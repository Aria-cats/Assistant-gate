import json
import os
import logging
import random
import time
from tqdm import tqdm
import requests

# 配置日志
logging.basicConfig(level=logging.INFO)

# 配置参数
SPLIT_NAME = "test"  # 默认值，可以是 "A"、"B" 或 "test"
API_KEY = "dfc707cb48024c5ab6342655fe4f04a8ac6da50922e84fe083c3c48f5646411f"  # 你的API密钥
MODEL = "gpt-3.5-turbo"  # 或 "gpt-4"
NUM_SAMPLES = 2  # 要处理的角色-任务组合数量

# 路径配置 - 与原始代码保持一致
# 使用绝对路径
VERSION_2_BSFT = "v2-bsft"
PERSONAS_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/persona-generation/outputs/{VERSION_2_BSFT}"
PROMPT_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/instruct-questions/outputs/{VERSION_2_BSFT}"
GOLD_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/build-gold-responses/outputs/{VERSION_2_BSFT}"
# 确保输出目录存在
os.makedirs(GOLD_PATH, exist_ok=True)

# 系统提示和生成提示模板 - 模拟原始代码中的内容
SYS_PROMPT = "You are a helpful, comprehensive, and accurate assistant. You are assisting a user with their request."
GENERATION_PROMPT = """You are answering questions for the following user:

{0}

Answer the question below, tailoring your answer to the user and their characteristics:

{1}"""

# 加载角色和任务数据
def load_data(split):
    """加载指定拆分的角色和任务数据"""
    try:
        # 加载角色数据
        personas_file = f"{PERSONAS_PATH}/{split}.json"
        logging.info(f"Loading personas from {personas_file}")
        with open(personas_file, "r") as f:
            personas = json.load(f)
        
        # 加载任务数据
        prompts_file = f"{PROMPT_PATH}/{split}.json"
        logging.info(f"Loading prompts from {prompts_file}")
        with open(prompts_file, "r") as f:
            prompts = json.load(f)
            prompts = [s.strip() for s in prompts]
        
        logging.info(f"Loaded {len(personas)} personas and {len(prompts)} prompts for split {split}")
        return personas, prompts
    except Exception as e:
        logging.error(f"Error loading data for split {split}: {e}")
        return [], []

# 生成Oracle响应
def generate_oracle_response(persona, prompt):
    """为给定的角色-提示组合生成Oracle响应"""
    url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 使用与原始代码相同的格式构建提示
    formatted_prompt = GENERATION_PROMPT.format(persona, prompt)
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": formatted_prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            logging.error(f"API error: {result}")
            return "Error generating response."
    except Exception as e:
        logging.error(f"Exception: {e}")
        return f"Error: {str(e)}"

# 将列表分成批次
def batch_list(lst, batch_size):
    """将列表分成指定大小的批次"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# 主函数
def main():
    # 确定要处理的拆分
    splits = [SPLIT_NAME]
    if SPLIT_NAME == "all":
        splits = ["A", "B", "test"]
    
    for split in splits:
        logging.info(f"Processing split: {split}")
        
        # 加载数据
        personas, prompts = load_data(split)
        if not personas or not prompts:
            logging.error(f"No data available for split {split}. Skipping.")
            continue
        
        # 限制数据量以节省API调用
        if len(personas) > 2:
            personas = personas[:2]  # 只使用前2个角色
        if len(prompts) > 5:
            prompts = prompts[:5]  # 只使用前5个提示
        
        # 初始化结果字典
        gold_responses = {}
        
        # 为每个角色和提示生成响应
        for j, persona in enumerate(personas):
            logging.info(f"Generating gold responses for persona {j}...")
            
            # 将提示分成批次
            batch_size = 2  # 小批量处理
            prompt_batches = list(batch_list(prompts, batch_size))
            
            for batch_index, prompt_batch in enumerate(prompt_batches):
                for i, prompt in enumerate(tqdm(prompt_batch, desc=f"Batch {batch_index+1}/{len(prompt_batches)}")):
                    # 生成响应
                    response = generate_oracle_response(persona, prompt)
                    
                    # 创建与原始代码相同格式的键
                    prompt_index = batch_index * batch_size + i
                    gold_responses_key = f"prompt-{prompt_index} persona-{j}"
                    
                    # 存储响应
                    if gold_responses_key not in gold_responses:
                        gold_responses[gold_responses_key] = []
                    gold_responses[gold_responses_key].append(response)
                    
                    # 避免速率限制
                    time.sleep(1)
                
                # 每批次完成后保存中间结果
                output_file = f"{GOLD_PATH}/{split}.json"
                with open(output_file, 'w') as f:
                    json.dump(gold_responses, f, indent=2)
                logging.info(f"Saved intermediate results to {output_file}")
        
        # 保存最终结果
        output_file = f"{GOLD_PATH}/{split}.json"
        with open(output_file, 'w') as f:
            json.dump(gold_responses, f, indent=2)
        
        logging.info(f"Saved {len(gold_responses)} gold responses to {output_file}")

if __name__ == "__main__":
    # 处理命令行参数
    import sys
    for arg in sys.argv[1:]:
        if arg.startswith('--split=') or arg.startswith('split='):
            SPLIT_NAME = arg.split('=')[1]
    
    random.seed(1)  # 设置随机种子，确保结果可重现
    logging.info(f"Starting gold response generation for split: {SPLIT_NAME}")
    main()