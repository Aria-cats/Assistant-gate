import json
import os
import logging
from datasets import load_dataset

# 配置日志
logging.basicConfig(level=logging.INFO)

# 设置输出目录 - 使用你有权限的目录
OUTPUT_DIR = "/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/instruct-questions/outputs/v2-bsft"

def main():
    logging.info("Extracting initial prompts...")
    
    # 提取初始提示的函数
    def extract_initial_prompt(conversation):
        try:
            # 找到"Assistant: "第一次出现的位置
            assistant_idx = conversation.find('Assistant: ')
            if assistant_idx == -1:
                return conversation.strip()
            
            # 提取从开始到"Assistant: "之前的部分
            initial_part = conversation[:assistant_idx].strip()
            
            # 去掉开头的"Human: "
            if initial_part.startswith('Human: '):
                initial_part = initial_part[len('Human: '):]
            
            return initial_part
        except Exception as e:
            logging.error(f"Error extracting prompt: {e}")
            return conversation
    
    try:
        # 尝试加载数据集
        logging.info("Loading dataset...")
        full_dataset = load_dataset("Dahoas/instruct-human-assistant-prompt", split='train').to_pandas()
        logging.info(f"Dataset loaded, shape: {full_dataset.shape}")
        
        # 定义拆分大小
        PUBLIC_ROWS = 250  # 每个公共拆分的行数
        PRIVATE_ROWS = 50  # 测试拆分的行数
        
        # A拆分 - 提取初始提示
        A_public_exchanges = full_dataset.iloc[:PUBLIC_ROWS]['prompt'].tolist()
        A_public_initial_turns = [extract_initial_prompt(exchange) for exchange in A_public_exchanges]
        logging.info(f"Extracted {len(A_public_initial_turns)} prompts for A split")
        
        # B拆分 - 提取初始提示
        B_public_exchanges = full_dataset.iloc[PUBLIC_ROWS:PUBLIC_ROWS * 2]['prompt'].tolist()
        B_public_initial_turns = [extract_initial_prompt(exchange) for exchange in B_public_exchanges]
        logging.info(f"Extracted {len(B_public_initial_turns)} prompts for B split")
        
        # 测试拆分 - 提取初始提示
        private_exchanges = full_dataset.iloc[PUBLIC_ROWS * 2:PUBLIC_ROWS * 2 + PRIVATE_ROWS]['prompt'].tolist()
        private_initial_turns = [extract_initial_prompt(exchange) for exchange in private_exchanges]
        logging.info(f"Extracted {len(private_initial_turns)} prompts for test split")
    
    except Exception as e:
        # 如果加载数据集失败，使用示例数据
        logging.error(f"Error loading dataset: {e}")
        logging.info("Using example prompts instead")
        
        A_public_initial_turns = [
            "Write a short story about a dragon.",
            "Explain quantum computing to a 10-year-old.",
            "Give me tips for improving my resume."
        ]
        
        B_public_initial_turns = [
            "What are some healthy breakfast options?",
            "How can I learn to play the guitar?",
            "Describe the process of photosynthesis."
        ]
        
        private_initial_turns = [
            "What are the benefits of meditation?",
            "How do I start a vegetable garden?",
            "Explain the theory of relativity."
        ]
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")
    
    # 保存提取的任务
    try:
        # A拆分
        with open(os.path.join(OUTPUT_DIR, "A.json"), 'w') as f:
            json.dump(A_public_initial_turns, f, indent=2)
            logging.info(f"Saved A split prompts to {os.path.join(OUTPUT_DIR, 'A.json')}")
        
        # B拆分
        with open(os.path.join(OUTPUT_DIR, "B.json"), 'w') as f:
            json.dump(B_public_initial_turns, f, indent=2)
            logging.info(f"Saved B split prompts to {os.path.join(OUTPUT_DIR, 'B.json')}")
        
        # 测试拆分
        with open(os.path.join(OUTPUT_DIR, "test.json"), 'w') as f:
            json.dump(private_initial_turns, f, indent=2)
            logging.info(f"Saved test split prompts to {os.path.join(OUTPUT_DIR, 'test.json')}")
        
        logging.info("All prompts saved successfully!")
    
    except Exception as e:
        logging.error(f"Error saving prompts: {e}")

if __name__ == "__main__":
    main()