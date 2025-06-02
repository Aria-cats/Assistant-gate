import json
import os
import logging
import random
import time
import requests
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)

# 配置参数
SPLIT_NAME = "test"  # 默认值，可以是 "A"、"B" 或 "test"
API_KEY = "dfc707cb48024c5ab6342655fe4f04a8ac6da50922e84fe083c3c48f5646411f"  # 你的API密钥
MODEL = "gpt-3.5-turbo"  # 或 "gpt-4"

# 路径配置
VERSION_2_BSFT = "v2-bsft"
PROMPT_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/instruct-questions/outputs/{VERSION_2_BSFT}"
PERSONAS_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/persona-generation/outputs/{VERSION_2_BSFT}"
SIMULATION_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/simulate-conversations/outputs/{VERSION_2_BSFT}"
GOLD_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/build-gold-responses/outputs/{VERSION_2_BSFT}"
LOGPROBS_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/log-probs/outputs/{VERSION_2_BSFT}"

# 确保输出目录存在
os.makedirs(f"{LOGPROBS_PATH}/qa/m0", exist_ok=True)

# 计算相似度分数（作为对数概率的替代）
def calculate_similarity(conversation, gold_response):
    """计算对话最终回答与gold响应的相似度"""
    url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 提取对话的最终部分
    final_response = ""
    
    # 处理不同类型的conversation
    if isinstance(conversation, list):
        final_response = conversation[-1] if len(conversation) > 0 else ""
    elif isinstance(conversation, dict):
        # 如果是字典，尝试获取最后一个回答
        # 打印字典结构以便调试
        logging.info(f"Dictionary structure: {conversation.keys()}")
        
        # 尝试获取常见的回答字段
        if "response" in conversation:
            final_response = conversation["response"]
        elif "assistant_response" in conversation:
            final_response = conversation["assistant_response"]
        elif "answers" in conversation and isinstance(conversation["answers"], list) and len(conversation["answers"]) > 0:
            final_response = conversation["answers"][-1]
        elif "messages" in conversation and isinstance(conversation["messages"], list):
            # 尝试从messages列表中获取最后一条助手消息
            assistant_messages = [msg["content"] for msg in conversation["messages"] 
                               if isinstance(msg, dict) and msg.get("role") == "assistant"]
            if assistant_messages:
                final_response = assistant_messages[-1]
            else:
                final_response = str(conversation)  # 转为字符串作为后备方案
        else:
            final_response = str(conversation)  # 转为字符串作为后备方案
    else:
        # 处理字符串类型
        try:
            conversation_parts = conversation.split("Assistant:")
            final_response = conversation_parts[-1].strip() if len(conversation_parts) > 1 else conversation
        except AttributeError:
            # 如果出现错误，记录错误并返回原始对象的字符串表示
            logging.error(f"Unexpected type for conversation: {type(conversation)}")
            final_response = str(conversation)
    
    prompt = f"""Please evaluate the similarity between the following two texts on a scale from 0 to 10, where 0 means completely different and 10 means identical in meaning.

Text 1:
{final_response}

Text 2:
{gold_response}

Provide only a numerical score without any explanation. Just the number."""
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that evaluates the similarity between texts."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1  # 低温度，使输出更确定性
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            # 尝试提取数字
            import re
            match = re.search(r'(\d+(\.\d+)?)', content)
            if match:
                score = float(match.group(1))
                return min(score, 10.0) / 10.0  # 标准化到0-1
            return 0.5  # 默认中等相似度
        else:
            logging.error(f"API error: {result}")
            return 0.5
    except Exception as e:
        logging.error(f"Exception during API call: {e}")
        return 0.5

# 主函数
def main():
    logging.info(f"Starting log probability calculation for split: {SPLIT_NAME}")
    
    try:
        # 加载prompts
        with open(f"{PROMPT_PATH}/{SPLIT_NAME}.json", "r") as f:
            prompts = json.load(f)
            prompts = [s.strip() for s in prompts]
        logging.info(f"Loaded {len(prompts)} prompts")
        
        # 加载personas
        with open(f"{PERSONAS_PATH}/{SPLIT_NAME}.json", "r") as f:
            personas = json.load(f)
        logging.info(f"Loaded {len(personas)} personas")
        
        # 尝试加载names（如果不存在，使用默认名称）
        try:
            with open(f"{PERSONAS_PATH}/{SPLIT_NAME}_NAMES.json", "r") as f:
                names = json.load(f)
        except:
            logging.info("Names file not found, using default names")
            names = [f"User {i+1}" for i in range(len(personas))]
        
        # 加载对话
        conversation_file = f"{SIMULATION_PATH}/m0/{SPLIT_NAME}.json"
        with open(conversation_file, "r") as f:
            all_conversations = json.load(f)
        logging.info(f"Loaded conversations from {conversation_file}")
        
        # 加载gold响应
        with open(f"{GOLD_PATH}/{SPLIT_NAME}.json", "r") as f:
            gold_responses = json.load(f)
        logging.info(f"Loaded gold responses")
        
        # 限制数据量以节省API调用
        max_prompts = 3
        max_personas = 2
        
        # 计算对数概率（这里用相似度替代）
        final_log_probs = defaultdict(list)
        
        # 处理有限数量的persona和prompt
        for j, persona in enumerate(personas[:max_personas]):
            for i, prompt in enumerate(prompts[:max_prompts]):
                logging.info(f"Computing similarity for prompt-{i} persona-{j}...")
                
                # 获取当前persona-prompt组合的对话
                key = f"prompt-{i} persona-{j}"
                if key not in all_conversations:
                    logging.warning(f"No conversations found for {key}")
                    continue
                
                conversations = all_conversations[key]
                
                # 获取gold响应
                if key not in gold_responses:
                    logging.warning(f"No gold response found for {key}")
                    continue
                
                gold_response = gold_responses[key][0]  # 使用第一个gold响应
                
                # 对每个对话计算相似度
                similarities = []
                for conv in tqdm(conversations, desc=f"Processing {key}"):
                    # 计算相似度
                    similarity = calculate_similarity(conv, gold_response)
                    similarities.append(similarity)
                    
                    # 避免速率限制
                    time.sleep(1)
                
                # 存储结果
                final_log_probs[key] = similarities
                
                # 每处理完一个组合就保存一次结果
                output_file = f"{LOGPROBS_PATH}/qa/m0/{SPLIT_NAME}.json"
                with open(output_file, "w") as f:
                    json.dump(final_log_probs, f, indent=2)
                logging.info(f"Saved intermediate results to {output_file}")
        
        # 保存最终结果
        output_file = f"{LOGPROBS_PATH}/qa/m0/{SPLIT_NAME}.json"
        with open(output_file, "w") as f:
            json.dump(final_log_probs, f, indent=2)
        
        logging.info(f"Saved log probabilities to {output_file}")
        
        # 计算平均相似度
        all_similarities = [sim for sims in final_log_probs.values() for sim in sims]
        avg_similarity = np.mean(all_similarities) if all_similarities else 0
        logging.info(f"Average similarity score: {avg_similarity:.4f}")
        
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置随机种子，确保结果可重现
    random.seed(1)
    main()