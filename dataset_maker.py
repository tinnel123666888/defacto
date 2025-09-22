import pandas as pd
import json
import os
from PIL import Image
from datasets import DatasetDict, Dataset, concatenate_datasets
from tqdm import tqdm
import multiprocessing
import numpy as np
import math
import glob
Image.MAX_IMAGE_PIXELS = 2800000000

def resize_image_and_adjust_boxes(image, solution, max_size=840):
    """Resize image and adjust bounding boxes accordingly"""
    # 获取原始尺寸
    orig_width, orig_height = image.size
    scale = max_size / max(orig_width, orig_height)
    if max(orig_width, orig_height) <=840:
        resized_image = image.copy()
        return resized_image,solution
    # 计算新尺寸
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # 解析solution中的bounding box
    try:
        # 提取answer部分
        answer_start = solution.find('<bbox>') + len('<bbox>')
        answer_end = solution.find('</bbox>')
        answer_str = solution[answer_start:answer_end]
        if 'nonebox' in answer_str:
            new_solution = solution
        else:
            # 转换为Python对象
            answer_data = eval(answer_str)
            
            # 调整每个bounding box的坐标
            for item in answer_data:
                pos = item['Position']
                
                # 调整坐标
                item['Position'] = [
                    int(pos[0] * scale),
                    int(pos[1] * scale),
                    int(pos[2] * scale),
                    int(pos[3] * scale)
                ]
                # print(pos,'------>',item['Position'])
            # 重新生成answer字符串
            new_answer_str = str(answer_data).replace("'", '"')
            new_solution = solution.replace(answer_str, new_answer_str)
        
    except Exception as e:
        print(f"Error processing bounding boxes: {e}")
        new_solution = solution
    
    return resized_image, new_solution

def process_item(args):
    item, max_size = args
    try:

        conversations = item["conversations"]
        user_content = ""
        assistant_answer = ""
        
        for conv in conversations:
            if conv["from"] == "user":
                user_content = conv["value"]
            elif conv["from"] == "assistant":
                assistant_answer = conv["value"]

        image_path = item["images"][0] if item.get("images") else None
        if 'original_masked' in image_path:
            return {
            'image': None,
            'problem': None,
            'solution': None,
            'status': False
        }
        problem = user_content.replace('<image>','')
        # solution = item['solution']
        solution = f"<bbox>[{{'Position': [0,0,0,0], 'Confidence': 1.0}}]</bbox><answer>{assistant_answer}</answer>"
        # 打开并处理图像
        image = Image.open(image_path).convert('RGB')
        # resized_image, adjusted_solution = resize_image_and_adjust_boxes(image, solution, max_size)
        resized_image,adjusted_solution = image.copy(),solution

        image.close()
        # print(problem)
        # print(adjusted_solution)
        # 返回处理结果（图像转换为数组以便进程间传递）
        return {
            'image': resized_image,
            'problem': problem,
            'solution': adjusted_solution,
            'status': True
        }
    except Exception as e:
        # print(f"Error processing {item.get('image_path', 'unknown')}: {e}")
        return {
            'image': None,
            'problem': None,
            'solution': None,
            'status': False
        }

def json_to_dataset(json_file_paths, max_size=840, num_processes=None, batch_size=25000):
    all_data = []
    # json_file_paths = [json_file_paths[0]]
    # 读取所有JSON文件
    for json_file_path in json_file_paths:
        if 'infer' in json_file_path:
            continue
        print(json_file_path)
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    # 设置进程数（默认为CPU核心数）
    if num_processes is None:
        num_processes = multiprocessing.cpu_count() - 1
    
    # 准备多进程参数
    task_args = [(item, max_size) for item in all_data]
    
    # 使用进程池处理
    successful_items = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_item, task_args),
            total=len(all_data),
            desc="Processing images"
        ))
        
        # 过滤成功处理的项
        for result in results:
            if result['status']:
                successful_items.append({
                    'image': result['image'],
                    'problem': result['problem'],
                    'solution': result['solution']
                })
    
    # 计算需要划分的批次数量
    num_batches = math.ceil(len(successful_items) / batch_size)
    
    # 创建并保存每个批次的数据集
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(successful_items))
        
        batch_items = successful_items[start_idx:end_idx]
        
        # 创建当前批次的数据集
        dataset_dict = {
            'image': [item['image'] for item in batch_items],
            'problem': [item['problem'] for item in batch_items],
            'solution': [item['solution'] for item in batch_items]
        }
        
        # 创建DatasetDict并保存
        batch_dataset = DatasetDict({'train': Dataset.from_dict(dataset_dict)})
        batch_save_path = f'/data1/zhu/xtr/train_new/batch_{batch_idx}'
        save_dataset(batch_dataset, batch_save_path)
        print(f"批次 {batch_idx} 已保存到: {batch_save_path}")
    
    return num_batches

def save_dataset(dataset_dict, save_path):
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    # 保存DatasetDict到磁盘
    dataset_dict.save_to_disk(save_path)

def load_dataset(save_path):
    # load DatasetDict
    return DatasetDict.load_from_disk(save_path)

# 使用示例
if __name__ == "__main__":
    # 指定包含JSON文件的文件夹路径
    folder_path = '/data1/zhu/xtr/dataset_all_json/train_new'
    
    # 获取文件夹内所有JSON文件
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    
    # 创建并保存调整后的数据集
    num_batches = json_to_dataset(json_files, batch_size=25000)
    print(f"总共保存了 {num_batches} 个批次的数据集")