import requests
import json
import os
import logging
import random
import time
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)

# 配置参数
SPLIT_NAME = "test"  # 默认值
API_KEY = "dfc707cb48024c5ab6342655fe4f04a8ac6da50922e84fe083c3c48f5646411f"  # 替换为你的API密钥
MODEL = "gpt-3.5-turbo"  # 或 "gpt-4"
OUTPUT_DIR = "experiments/star-gate/persona-generation/outputs/v2-bsft"
NUM_PERSONAS = 3  # 要生成的角色数量

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 系统消息
SYSTEM_MESSAGE = "You are a helpful assistant that generates diverse and unique user personas."

# 从GitHub获取示例角色
def get_expert_personas(limit=3):
    url = 'https://raw.githubusercontent.com/LanD-FBK/prodigy-dataset/main/dataset/characters.json'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text.strip()
            data = '{' + data[1:len(data) - 1] + '}'
            data = json.loads(data)
            
            # 只使用几个角色作为示例
            keys = list(data.keys())[:limit]
            filtered_data = {k: data[k] for k in keys if k in data}
            
            expert_personas = [f"I'm {char_data['character_name']}. {' '.join(char_data['biography'])}" 
                              for _, char_data in filtered_data.items()]
            return expert_personas
        else:
            logging.error(f"Failed to retrieve personas: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Error getting expert personas: {e}")
        return []

# 调用学校API生成角色
def generate_persona(prompt, system_message):
    url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            logging.error(f"API error: {result}")
            return "Error generating persona."
    except Exception as e:
        logging.error(f"Exception: {e}")
        return f"Error: {str(e)}"

# 主函数
def main():
    # 获取专家角色示例
    logging.info("Fetching expert personas...")
    expert_personas = get_expert_personas(limit=3)
    
    if not expert_personas:
        logging.error("No expert personas available. Using default examples.")
        expert_personas = [
            "I'm Alex, a 35-year-old software engineer with a passion for problem-solving and efficiency.",
            "I'm Jordan, a creative 28-year-old graphic designer who values aesthetics and clarity."
        ]
    
    # 生成提示模板
    base_prompt = """Create a detailed persona for an individual with unique preferences, background, and communication style. 
The persona should be written in first person, starting with 'I'm [Name]'.

Here are some examples of well-written personas:

{}

Now create a new, unique persona that is different from these examples."""
    
    # 为每个拆分生成角色
    splits = [SPLIT_NAME]
    if SPLIT_NAME == "all":
        splits = ["A", "B", "test"]
    
    for split in splits:
        logging.info(f"Generating {NUM_PERSONAS} personas for split '{split}'...")
        personas = []
        
        for i in tqdm(range(NUM_PERSONAS)):
            # 随机选择2个示例
            examples = random.sample(expert_personas, min(2, len(expert_personas)))
            examples_text = "\n\n".join(examples)
            
            prompt = base_prompt.format(examples_text)
            persona = generate_persona(prompt, SYSTEM_MESSAGE)
            personas.append(persona)
            
            # 避免速率限制
            time.sleep(1)
        
        # 保存结果
        output_file = os.path.join(OUTPUT_DIR, f"{split}.json")
        with open(output_file, 'w') as f:
            json.dump(personas, f, indent=2)
        
        logging.info(f"Successfully saved {len(personas)} personas to {output_file}")

if __name__ == "__main__":
    # 处理命令行参数
    import sys
    for arg in sys.argv[1:]:
        if arg.startswith('--split=') or arg.startswith('split='):
            SPLIT_NAME = arg.split('=')[1]
    
    logging.info(f"Starting persona generation for split: {SPLIT_NAME}")
    main()