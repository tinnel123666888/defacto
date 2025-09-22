# inference_client.py
import requests
from PIL import Image, ImageDraw, ImageFont
import os
import time
import json
import concurrent.futures
from tqdm import tqdm
import threading

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
    """Send image to API and process results"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
        
    for attempt in range(max_retries):
        try:
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                data = {'question': question}
                
                response = requests.post('http://localhost:5000/detect', data=data, files=files, timeout=300)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get('status') == 'error':
                    print(f"API returned error: {result.get('error', 'Unknown error')}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                
                return result
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"Max retries exceeded for {image_path}")
                return None
        except Exception as e:
            print(f"Unexpected error processing {image_path}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None
    return None

def process_item(item, lock, results, index):
    """Process a single item and store the result"""
    image_path = item.get('image_path')
    question = item.get('problem', '').replace('<image>', '')
    question = '<image>' + question
    
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
            'boxx': bbox_positions
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
                'boxx': [],
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
                                'solution':data[index].get('solution'),
                                'boxx': [],
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
    input_json_path = "/data1/zhu/xtr/Grpo_train/new_data.json"  # 替换为您的输入JSON路径
    output_json_path = "/data1/zhu/xtr/Grpo_train/output.json"  # 替换为您的输出JSON路径
    max_workers = 8  # 根据您的API承受能力调整并发数
    checkpoint_interval = 1000  # 每处理1000张图保存一次
    
    process_json_file(input_json_path, output_json_path, max_workers, checkpoint_interval)