import os
os.environ["WANDB_API_KEY"] = 'c28104020dd1f6ab7326cd2128901ee7c4398246'

os.environ["WANDB_MODE"] = "offline"
import re
from difflib import SequenceMatcher
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from rapidfuzz import fuzz
from datasets import load_dataset, load_from_disk,concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration

# from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )




def extract_bbox(response):
    start_tag = "<bbox>"
    end_tag = "</bbox>"
    input_str = response
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
    
        # Check if it ends with a closing bracket, if not, fix it
        if not content_str.endswith("]"):
            # If the string is truncated, remove the incomplete part
            content_str = content_str.rsplit("},", 1)[0] + "}]"
    
        # Replace single quotes with double quotes for valid JSON
        content_str_corrected = content_str.replace("'", '"')
    
        # Convert the corrected string to a list of dictionaries (JSON format)
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError as e:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou

def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    
    iou_results = []
    
    matched_list1_indices = set()

    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:
                
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
        else:
            iou_results.append((0, bbox2['Confidence']))
    
    ### [(0.7192676547515258, 1.0), (0, 0.7)]
    return iou_results

def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['Position'])
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes

# V1
def compute_reward_iou(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return iou_reward

# V2
def compute_reward_iou_v2(iou_results, len_gt):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    if len_gt>=len(iou_results):
        iou_reward = iou_reward/len_gt
    else:
        iou_reward = iou_reward/len(iou_results)
    return iou_reward

def accuracy_reward_iou(completions, solution, pos_box, neg_box,ids, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol, pos_box, neg_box,id_type in zip(contents, solution, pos_box, neg_box,ids):
        reward = 0.0
        try:
            # Convert pos_box and neg_box to the expected format
            pos_bboxes = [{'Position': box, 'Confidence': 1.0} for box in pos_box] if pos_box else []
            neg_bboxes = [{'Position': box, 'Confidence': 1.0} for box in neg_box] if neg_box else []
            
            if id_type == 'neg':
                # Negative sample: reward -1 for any bbox, reward 1 for no bbox or unknown
                if 'unknown' in content.lower():
                    reward = 1.0
                else:
                    # Try to extract bbox from content
                    content_match = re.search(r'<bbox>(.*?)</bbox>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()
                    student_answer = '<bbox>' + student_answer + '</bbox>'
                    
                    # Fix format errors
                    student_answer = student_answer.replace("[[", '[')
                    student_answer = student_answer.replace("]]", ']')
                    student_answer = student_answer.replace("\n", '')
                    
                    student_answer_bbox = extract_bbox(student_answer)
                    if student_answer_bbox and len(student_answer_bbox) > 0:
                        reward = -1.0
                    else:
                        reward = 1.0
            else:  # pos or random
                # Positive or random sample: calculate IoU with pos_box and neg_box
                content_match = re.search(r'<bbox>(.*?)</bbox>', content)
                if not content_match:
                    reward = 0.0
                    rewards.append(reward)
                    continue
                    
                student_answer = content_match.group(1).strip()
                student_answer = '<bbox>' + student_answer + '</bbox>'
                
                # Fix format errors
                student_answer = student_answer.replace("[[", '[')
                student_answer = student_answer.replace("]]", ']')
                student_answer = student_answer.replace("\n", '')
                
                student_answer_bbox = extract_bbox(student_answer)
                if not student_answer_bbox or len(student_answer_bbox) == 0:
                    reward = 0.0
                    rewards.append(reward)
                    continue
                
                # Remove duplicates from student answer
                student_answer_bbox = remove_duplicates(student_answer_bbox)
                
                # Calculate IoU with pos_boxes (positive reward)
                pos_iou_results = sort_and_calculate_iou(pos_bboxes, student_answer_bbox)
                pos_reward = compute_reward_iou_v2(pos_iou_results, len(pos_bboxes)) if pos_bboxes else 0.0
                
                # Calculate IoU with neg_boxes (negative reward)
                neg_iou_results = sort_and_calculate_iou(neg_bboxes, student_answer_bbox)
                neg_reward = compute_reward_iou_v2(neg_iou_results, len(neg_bboxes)) if neg_bboxes else 0.0
                
                # Final reward is positive IoU minus negative IoU
                reward = pos_reward - neg_reward
                reward = max(-1.0, min(1.0, reward))  # Clamp between -1 and 1
                
        except Exception as e:
            print(f'Error: {e}')
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

def compute_reward_confidence(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return confidence_reward

import re
import os
from difflib import SequenceMatcher

def accuracy_reward_text(completions, solution, pos_box, neg_box,ids, **kwargs):
    """Reward function that checks if the completion is correct using exact string matching or text similarity."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # Define negative response patterns (case insensitive)
    negative_patterns = [r'unknown', r'no', r'none']
    negative_regex = re.compile('|'.join(negative_patterns), re.IGNORECASE)
    
    for content, sol, id_type in zip(contents, solution, ids):
        reward = 0.0
        show_flage = 0
        
        # Check if response contains negative patterns
        has_negative_response = bool(negative_regex.search(content))
        
        if id_type in ['pos', 'random']:
            if has_negative_response:
                reward = -1.0
            else:
                try:
                    # Extract answer from solution if it has answer tags
                    sol_match = sol.strip().lower()
                    if sol_match:
                        ground_truth = sol_match
                    else:
                        ground_truth = ""

                    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                    if content_match:
                        student_answer = content_match.group(1).strip().lower()
                    else:
                        student_answer = content.strip().lower()
                    
                    # Exact match check
                    if ground_truth == student_answer:
                        reward = 1.0
                    else:
                        reward = 0.0
                except Exception as e:
                    print('############', e)
                    reward = 0.0
        
        elif id_type == 'neg':
            reward = 1.0 if has_negative_response else -1.0

        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true" and show_flage == 1:
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"ground_truth: {ground_truth}\n")
                f.write(f"student_answer: {student_answer}\n")
    
    return rewards




def think_format_reward(completions, **kwargs):
    """Reward function that checks if the <think> tag contains at least 10 characters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        # 使用正则表达式提取 <think> 标签内的内容
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        
        if think_match:
            think_content = think_match.group(1).strip()  # 获取标签内的文本并去除首尾空白
            # 检查字符数是否 >=10
            reward = 1.0 if len(think_content) >= 10 else 0.0
        else:
            reward = 0.0  # 没有找到 <think> 标签
        rewards.append(reward)
    
    return rewards

def box_format_reward(completions, **kwargs):
    """Reward function that checks if the <think> tag contains at least 10 characters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        # 使用正则表达式提取 <think> 标签内的内容
        think_match = re.search(r"<bbox>(.*?)</bbox>", content, re.DOTALL)
        
        if think_match:
            think_content = think_match.group(1).strip()  # 获取标签内的文本并去除首尾空白
            # 检查字符数是否 >=10
            reward = 1.0 if len(think_content) >= 10 else 0.0
        else:
            reward = 0.0  # 没有找到 <think> 标签
        rewards.append(reward)
    
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<bbox>.*?</bbox>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    print(completion_contents)
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

###  reward registry three parts
reward_funcs_registry = {
    "accuracy_iou": accuracy_reward_iou,
    # "accuracy_confidence": accuracy_reward_confidence,
    "format": format_reward,'accuracy_reward_text':accuracy_reward_text,"think_format":think_format_reward,"box_format":box_format_reward}


SYSTEM_PROMPT = "Please answer my question based on the image I have provided. Identify the region in the image that is most relevant to the question, provide the bounding box and confidence (between 0 and 1, with two decimal places).  Output the bounding boxes within <bbox> </bbox> tags, and the thinking process in <think> </think> tags,and the final answer within <answer> </answer> tags. The format of the response should strictly follow the structure below:  <bbox>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...] </bbox><think> ... </think><answer>xxxxxxxx</answer>.If there are multiple relevant regions, please provide multiple bounding boxes. If no relevant information is present in the image, respond with 'unknown' within <bbox> </bbox> tags also <answer> </answer> tags.The current question is:"

def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['format','accuracy_reward_text','accuracy_iou']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    from datasets import DatasetDict
    train_sets = []
    dataset_tmp = {}
    import glob
    for i in range(len(glob.glob(script_args.dataset_name + '/batch_*'))):
        path = script_args.dataset_name + f'/batch_{i}'
        dataset_tmp[f'dataset{i}'] = DatasetDict.load_from_disk(path)
        train_sets.append(dataset_tmp[f'dataset{i}']["train"])
    con_dataset = concatenate_datasets(train_sets)
    con_dataset =con_dataset.shuffle(seed=42)
    # con_dataset= con_dataset.select(range(120000))
    dataset = DatasetDict({
        "train": con_dataset
    })
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": SYSTEM_PROMPT + example["problem"]},
                    ],
                },
            ],
        }
    
    # 创建数据集
    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        
    else:
        print("no image in dataset")
        

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    main(script_args, training_args, model_args)
