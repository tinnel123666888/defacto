# inference_client.py
import requests
from PIL import Image, ImageDraw, ImageFont
import os
import time
import json
def resize_image(image, max_size=840):
    """Resize image while maintaining aspect ratio"""

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
    
def draw_results(image_path, result, output_path="detection_result.jpg"):
    """Draw bounding boxes and plate numbers on image"""
    # try:
    # 打开原始图像
    original_img = Image.open(image_path)
    # original_img = resize_image(original_img)
    w, h = original_img.size
    
    # 创建绘图对象
    draw = ImageDraw.Draw(original_img)
    
    # 使用更醒目的颜色和字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 绘制每个检测框
    for bbox in result.get('bboxes', []):
        pos = bbox['Position']
        conf = bbox['Confidence']
        print('pos',pos)
        # 绘制矩形
        draw.rectangle(pos, outline="green", width=3)
        
        # 添加置信度标签
        label = f"{pos}{conf:.2f}"
        draw.text((pos[0], pos[1] - 25), label, fill="green", font=font)
    
    
    # answer = result.get('answer', '')
    # if answer:
    #     plate_text = answer
    #     draw.rectangle([10, 10, 10 + len(plate_text)*10, 40], fill="black")
    #     draw.text((15, 15), plate_text, fill="red", font=font)
    
    # 保存结果
    original_img.save(output_path)
    print(f"Detection results saved to: {output_path}")
    return True
    # except Exception as e:
    #     print(f"Error drawing results: {str(e)}")
    #     return False

def detect_license_plate(image_path,question, max_retries=1, retry_delay=2):
    """Send image to API and process results"""
    # 1. 准备图像文件
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
        
    # try:
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {'question': question}
        # 2. 发送请求到API（带重试机制）
        for attempt in range(max_retries):
            try:
                print(f"Sending request to API (attempt {attempt+1}/{max_retries})...")
                response = requests.post('http://localhost:5000/detect', data = data,files=files, timeout=300)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get('status') == 'error':
                    print(f"API returned error: {result.get('error', 'Unknown error')}")
                    return None
                
                # 3. 显示结果
                print("\nAPI Response Summary:")
                
                if result.get('bboxes'):
                    print(f"Detected {len(result['bboxes'])} license plate(s):")
                    for i, bbox in enumerate(result['bboxes']):
                        pos = bbox['Position']
                        conf = bbox['Confidence']
                        print(f"  Plate {i+1}: Position={pos}, Confidence={conf:.2f}")
                else:
                    print("No license plates detected")
                
                if result.get('answer'):
                    plates = result['answer']
                    print(f"answer: {plates}")
                else:
                    print("No answer extracted")
                
                # 4. 绘制结果
                output_path = f"result_{os.path.basename(image_path)}"
                draw_success = draw_results(image_path, result, output_path)
                
                if draw_success:
                    print(f"Results saved to {output_path}")
                
                return result
                
            except requests.exceptions.RequestException as e:
                print(f"API request failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries exceeded")
                    return None
                        
    # except Exception as e:
    #     print(f"Unexpected error: {str(e)}")
    #     return None

if __name__ == '__main__':
    # 使用示例
    image_path = "/data1/zhu/xtr/9241757305421_.pic.jpg"
    prompt='Please answer my questions based on the pictures I provided. Determine the area in the image that is most relevant to the problem, and provide bounding boxes and confidence levels (between 0 and 1, with two decimal points). Output the bounding box within the <bbox> </bbox> tag, and then think based on the bounding box. Output the thinking process within the <think> </think> tag, as well as the final answer within the <answer> </answer> tag. The format of the reply should strictly follow the following structure: <bbox>[{'Position': [x1, y1, x2, y2], 'Confidence': number}]</bbox><think>...</think><answer>...</answer>. If there are multiple related areas, please provide multiple bounding boxes. The current problem is: '
    question =prompt+ 'How safe is this with no helmet especially for the passengers?'
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please verify the path or use a different image.")
        exit(1)
    
    print(f"Processing image: {image_path}")
    result = detect_license_plate(image_path,question)
    
    if result:
        # 可选：保存完整的API响应
        with open("api_response.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Full API response saved to api_response.json")
