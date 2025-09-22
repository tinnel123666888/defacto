# model_server.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    from .qwen_vl_utils import process_vision_info
from flask import Flask, request, jsonify
from PIL import Image
import io
import re
import json
import logging
import traceback

# 配置详细日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

SYSTEM_PROMPT = (
    "You are given a question and an image. Your task is to locate the region in the image that is most relevant for answering the question"
)

# 初始化模型和处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
model_path = '/data1/zhu/xtr/checkpoint-345'
try:
    logger.info("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    logger.info("Model loaded successfully")
    
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path
    )
    logger.info("Processor loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading model/processor: {str(e)}")
    logger.error(traceback.format_exc())
    # 尝试使用CPU作为后备
    if "cuda" in device:
        logger.info("Trying to load on CPU...")
        device = "cpu"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="cpu",
            torch_dtype=torch.float32,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_path
        )

def resize_image(image, max_size=840):
    """Resize image while maintaining aspect ratio"""
    try:
        w, h = image.size
        if max(w, h) <= max_size:
            return image
            
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
            
        return image.resize((new_w, new_h), Image.LANCZOS)
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image  # 返回原始图像作为后备

def prepare_inputs(image,question):
    """Prepare model inputs from PIL image"""
    # print(question)
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please answer my question based on the image I have provided. Identify the region in the image that is most relevant to the question and provide bounding box coordinates.  Output the thinking process in <think> </think> tags, the bounding boxes within <boxx> </boxx> tags, and the final answer within <answer> </answer> tags. The format of the response should strictly follow the structure below:  <think> ... </think><boxx>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...] </boxx><answer>xxxxxxxx</answer>.If there are multiple relevant regions, please provide multiple bounding boxes. If no relevant information is present in the image, respond with 'unknown' within <boxx> </boxx> tags also <answer> </answer> tags.The current question is:" + f'{question}'}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(device)
    except Exception as e:
        logger.error(f"Error preparing inputs: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def extract_bbox(response):
    """Extract bounding box from model response"""
    try:
        start_tag = "<boxx>"
        end_tag = "</boxx>"
        
        if start_tag in response:
            start_idx = response.find(start_tag) + len(start_tag)
            end_idx = response.find(end_tag)
            if end_idx == -1:
                end_idx = len(response)
        
            content_str = response[start_idx:end_idx].strip()
            content_str = content_str.replace("[[", '[').replace("]]", ']')
            content_str = content_str.replace("\n", "").replace("'", '"')
            print('response',response)
            print('content_str',content_str)
            # 尝试解析JSON
            try:
                return json.loads(content_str)
            except json.JSONDecodeError:
                # 使用更健壮的正则表达式
                pattern = r'\{[^{}]*?\}'
                matches = re.findall(pattern, content_str)
                bbox_list = []
                
                for match in matches:
                    # 提取位置
                    pos_match = re.search(r'Position\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', match)
                    # 提取置信度
                    conf_match = re.search(r'Confidence\s*:\s*([\d.]+)', match)
                    
                    if pos_match and conf_match:
                        x1, y1, x2, y2 = map(int, pos_match.groups())
                        conf = float(conf_match.group(1))
                        bbox_list.append({
                            "Position": [x1, y1, x2, y2],
                            "Confidence": conf
                        })
                return bbox_list
        return None
    except Exception as e:
        logger.error(f"Error extracting bbox: {str(e)}")
        return []

def extract_plate_number(response):
    """Extract plate number from model response"""
    try:
        start_tag = "<answer>"
        end_tag = "</answer>"
        
        if start_tag in response:
            start_idx = response.find(start_tag) + len(start_tag)
            end_idx = response.find(end_tag)
            if end_idx == -1:
                end_idx = len(response)
            
            content_str = response[start_idx:end_idx].strip()
            # content_str = content_str.replace("[[", '[').replace("]]", ']')
            # content_str = content_str.replace("\n", "").replace("'", '"')
            return content_str

            # data = json.loads(content_str)
            # if isinstance(data, list):
            #     return data
            # elif isinstance(data, dict):
            #     return [data]
        return []
    except Exception as e:
        logger.error(f"Error extracting plate number: {str(e)}")
        return []

@app.route('/')
def index():
    return "License Plate Detection API is running"

@app.route('/detect', methods=['POST'])
def detect_objects():
    """API endpoint for license plate detection"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        # 1. 接收并处理图像
        image_file = request.files['image']
        question = request.form.get('question')
        image_bytes = image_file.read()
        
        if len(image_bytes) == 0:
            return jsonify({"error": "Empty image file"}), 400
            
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400
        
        logger.info(f"Received image: {image_file.filename}, size: {image.size}")
        
        # 2. 调整图像大小
        resized_img = image
        # resized_img = resize_image(image)
        logger.info(f"Resized image to: {resized_img.size}")
        
        # 3. 准备模型输入
        inputs = prepare_inputs(resized_img,question)
        
        # 4. 运行模型推理
        logger.info("Running model inference...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=2048,
                use_cache=True,
                do_sample = False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]  # 切片操作：从输入序列结束位置开始截取
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]    
        response = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        print('---------')
        print(response)
        print('---------')
        logger.info(f"Model response: {response[:200]}...")  # 只记录前200个字符
        
        # 5. 解析结果
        bboxes = extract_bbox(response) or []
        answer = extract_plate_number(response) or []
        
        logger.info(f"Detected {len(bboxes)} bounding boxes and {len(answer)} plate numbers")
        
        return jsonify({
            "status": "success",
            "bboxes": bboxes,
            "answer": answer,
            "image_size": resized_img.size
        })
    
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # 测试模型加载
    try:
        test_input = processor(text=["Test"], return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.generate(**test_input, max_new_tokens=5)
        logger.info("Model test passed")
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
    
    # 启动服务
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)