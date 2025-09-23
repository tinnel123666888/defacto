# inference_client.py
import base64
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import os
import time
import json
import concurrent.futures
from tqdm import tqdm
import threading

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8111/v1"
SYSTEM_PROMPT = "Please answer my question based on the image I have provided. Identify the region in the image that is most relevant to the question and provide bounding box coordinates.  Output the thinking process in <think> </think> tags, the bounding boxes within <bbox> </bbox> tags, and the final answer within <answer> </answer> tags. The format of the response should strictly follow the structure below:  <think> ... </think><bbox>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...] </bbox><answer>xxxxxxxx</answer>.If there are multiple relevant regions, please provide multiple bounding boxes. If no relevant information is present in the image, respond with 'unknown' within <bbox> </bbox> tags also <answer> </answer> tags.The current question is:"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

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

def detect_license_plate(image_path, question, max_retries=3, retry_delay=2):
    """Send image to OpenAI API and process results"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
        
    for attempt in range(max_retries):
        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare the message content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": SYSTEM_PROMPT + question},
                    ],
                }
            ]
            
            # Call the OpenAI API
            chat_response = client.chat.completions.create(
                model="/data1/zhu/xtr/checkpoint-345",
                messages=messages,
                timeout=300
            )
            
            # Extract the response
            answer = chat_response.choices[0].message.content
            
            # For Qwen VL models, we need to parse the response to extract bounding boxes
            # This is a placeholder - you'll need to adjust this based on your model's response format
            bboxes = []
            if "bbox" in answer.lower() or "bounding box" in answer.lower():
                # Parse the bounding box information from the response
                # This is highly dependent on how your model formats its response
                # You might need to implement custom parsing logic here
                pass
                
            return {
                'answer': answer,
                'bboxes': bboxes
            }
                
        except Exception as e:
            print(f"API request failed (attempt {attempt+1}/{max_retries}) for {image_path}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"Max retries exceeded for {image_path}")
                return None
    return None

def process_item(item, lock, results, index):
    """Process a single item and store the result"""
    image_path = item.get('image_path')
    question = item.get('problem', '').replace('<image>', '')
    
    if not image_path:
        print(f"Skipping item due to missing image_path: {item}")
        with lock:
            results[index] = {'error': 'Missing image_path'}
        return
    
    result = detect_license_plate(image_path, question)
    
    if result:
        # Extract bounding boxes positions
        bbox_positions = []
        if result.get('bboxes'):
            for bbox in result['bboxes']:
                bbox_positions.append(bbox['Position'])
        
        # Prepare the result
        processed_item = {
            'image_path': image_path,
            'problem': item.get('problem'),
            'bbox': bbox_positions,
            'json': item.get('json')
        }
        
        # Also add answer if available
        if result.get('answer'):
            processed_item['answer'] = result['answer']
        
        with lock:
            results[index] = processed_item
    else:
        print(f"Failed to process image: {image_path}")
        with lock:
            results[index] = {
                'image_path': image_path,
                'problem': item.get('problem'),
                'bbox': [],
                'json': item.get('json'),
                'error': 'Processing failed after retries'
            }

def process_json_file(input_json_path, output_json_path, max_workers=10, checkpoint_interval=1000):
    """Process all items in the JSON file with concurrency and checkpointing"""
    # Read input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Check if output file exists to resume from last checkpoint
    start_index = 0
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f:
                existing_results = json.load(f)
                start_index = len(existing_results)
                print(f"Resuming from index {start_index}")
        except:
            print("Output file exists but could not be read. Starting from scratch.")
            existing_results = []
    else:
        existing_results = []
    
    # Initialize results list
    results = [None] * len(data)
    
    # Copy existing results if resuming
    for i in range(min(start_index, len(data))):
        results[i] = existing_results[i]
    
    # Create a lock for thread-safe operations
    lock = threading.Lock()
    
    # Process remaining items with concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_item, data[i], lock, results, i): i 
            for i in range(start_index, len(data))
        }
        
        # Process completed tasks with progress bar
        completed = 0
        with tqdm(total=len(future_to_index), desc="Processing images") as pbar:
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()  # This will re-raise any exceptions
                except Exception as e:
                    print(f"Item {index} generated an exception: {e}")
                    with lock:
                        if results[index] is None:
                            results[index] = {
                                'image_path': data[index].get('image_path'),
                                'problem': data[index].get('problem'),
                                'solution': data[index].get('solution'),
                                'bbox': [],
                                'error': f"Exception: {str(e)}"
                            }
                
                completed += 1
                pbar.update(1)
                
                # Save checkpoint every checkpoint_interval items
                if completed % checkpoint_interval == 0:
                    # Filter out None values (not yet processed)
                    current_results = [r for r in results if r is not None]
                    
                    # Save to a temporary file first
                    temp_path = output_json_path + ".tmp"
                    with open(temp_path, 'w') as f:
                        json.dump(current_results, f, indent=2)
                    
                    # Atomically replace the old file
                    os.replace(temp_path, output_json_path)
                    print(f"Checkpoint saved at {completed} items")
    
    # Final save
    # Filter out None values (shouldn't be any at this point)
    final_results = [r for r in results if r is not None]
    
    with open(output_json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Processing complete. Results saved to: {output_json_path}")

if __name__ == '__main__':
    input_json_path = "/data1/zhu/xtr/Grpo_train/new_data3.json"  # 替换为您的输入JSON路径
    output_json_path = "/data1/zhu/xtr/Grpo_train/output3.json"  # 替换为您的输出JSON路径
    max_workers = 16  # 根据您的API承受能力调整并发数
    checkpoint_interval = 400# 每处理1000张图保存一次
    
    process_json_file(input_json_path, output_json_path, max_workers, checkpoint_interval)
