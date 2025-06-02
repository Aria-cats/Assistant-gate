import json
import os
import logging
import random
import time
import requests
from tqdm import tqdm
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)

# 配置参数
SPLIT_NAME = "test"  # 默认值，可以是 "A"、"B" 或 "test"
API_KEY = "dfc707cb48024c5ab6342655fe4f04a8ac6da50922e84fe083c3c48f5646411f"  # 你的API密钥
MODEL = "gpt-3.5-turbo"  # 使用你学校API支持的模型
MAX_TURNS = 3  # 对话中的最大轮次数
NUM_RETURN_SEQUENCES = 1  # 每个提示生成的回答数量

# 路径配置 - 使用绝对路径
VERSION_2_BSFT = "v2-bsft"
PERSONAS_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/persona-generation/outputs/{VERSION_2_BSFT}"
PROMPT_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/instruct-questions/outputs/{VERSION_2_BSFT}"
SIMULATION_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate/simulate-conversations/outputs/{VERSION_2_BSFT}"

# 确保输出目录存在
os.makedirs(f"{SIMULATION_PATH}/m0", exist_ok=True)

# 提示模板
QA_PROMPT = "You are a helpful assistant having a conversation with {0}. Given their task: {1}, ask a clarifying question to better understand their needs before providing a full response. Your question should be specific and directly related to the task."

HUMAN_PROMPT = """You are roleplaying as the following person:

{0}

You've asked the assistant for help with the following task:
{1}

Previous conversation:
{2}

Now respond to the assistant's question in a way that reflects your persona."""

FINAL_RESPONSE_PROMPT = """You are a helpful assistant. Based on the conversation history, provide a final, comprehensive response to the original task.

Original task: {0}

Conversation history:
{1}

Now provide your detailed response to satisfy the user's original request, incorporating the information gathered during the conversation."""

# 辅助函数
def batch_list(lst, batch_size):
    """将列表分成指定大小的批次"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def flatten_list(list_of_lists):
    """扁平化嵌套列表"""
    return [item for sublist in list_of_lists for item in sublist]

def chunk_list(lst, chunk_size):
    """将列表分成指定大小的块"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def extract_history(conversation):
    """从对话文本中提取历史"""
    # 简化版：假设历史是对话的一部分
    return conversation.split('\n\n')[-1] if '\n\n' in conversation else ""

# API调用函数
def call_api(prompt, system_message="You are a helpful assistant."):
    """调用学校API生成文本"""
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
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            logging.error(f"API error: {result}")
            return "Error generating content."
    except Exception as e:
        logging.error(f"Exception during API call: {e}")
        return f"Error: {str(e)}"

# 模拟对话函数
def simulate_conversation(persona_name, persona, prompt, max_turns=MAX_TURNS):
    """模拟完整的对话过程"""
    conversation = []
    
    # 初始问题 (模型作为问答者)
    initial_qa_prompt = QA_PROMPT.format(persona_name, prompt)
    question = call_api(initial_qa_prompt, "You are a helpful assistant that asks clarifying questions.")
    conversation.append(f"Assistant: {question}")
    
    for turn in range(max_turns):
        # 角色回答 (模型作为角色扮演者)
        history = "\n".join(conversation)
        human_response_prompt = HUMAN_PROMPT.format(persona, prompt, history)
        human_response = call_api(human_response_prompt, "You are roleplaying as the described persona.")
        conversation.append(f"User: {human_response}")
        
        # 如果达到最大轮次，就停止提问
        if turn == max_turns - 1:
            break
        
        # 模型继续提问
        qa_prompt = f"You are having a conversation with a user about their task: '{prompt}'. So far, the conversation has been:\n\n{history}\n\nUser: {human_response}\n\nAsk another clarifying question to better understand their needs."
        next_question = call_api(qa_prompt, "You are a helpful assistant that asks clarifying questions.")
        conversation.append(f"Assistant: {next_question}")
    
    # 生成最终响应
    history = "\n".join(conversation)
    final_prompt = FINAL_RESPONSE_PROMPT.format(prompt, history)
    final_response = call_api(final_prompt, "You are a helpful assistant providing a final, comprehensive response.")
    
    # 将最终响应添加到对话中
    conversation.append(f"Assistant (Final Response): {final_response}")
    
    return {
        "task": prompt,
        "persona": persona,
        "conversation": conversation,
        "final_response": final_response
    }

# 主函数
def main():
    try:
        # 加载数据
        # 加载任务数据
        with open(f"{PROMPT_PATH}/{SPLIT_NAME}.json", "r") as f:
            prompts = json.load(f)
            prompts = [s.strip() for s in prompts]
            logging.info(f"Loaded {len(prompts)} prompts")
        
        # 加载角色数据
        with open(f"{PERSONAS_PATH}/{SPLIT_NAME}.json", "r") as f:
            personas = json.load(f)
            logging.info(f"Loaded {len(personas)} personas")
        
        # 尝试加载角色名称（如果不存在，使用默认名称）
        try:
            with open(f"{PERSONAS_PATH}/{SPLIT_NAME}_NAMES.json", "r") as f:
                names = json.load(f)
        except:
            logging.info("No names file found, using default names")
            names = [f"User {i+1}" for i in range(len(personas))]
        
        # 限制数据量以节省API调用
        if len(prompts) > 3:
            prompts = prompts[:3]
        if len(personas) > 2:
            personas = personas[:2]
            names = names[:2] if len(names) > 2 else names
        
        # 存储所有对话
        all_conversations = defaultdict(list)
        
        # 为每个角色-任务组合模拟对话
        for j, (persona, name) in enumerate(zip(personas, names)):
            logging.info(f"Simulating conversations for persona {j}: {name}")
            
            for i, prompt in enumerate(prompts):
                logging.info(f"  Task {i+1}/{len(prompts)}: {prompt[:50]}...")
                
                # 模拟对话
                conversation_data = simulate_conversation(name, persona, prompt)
                
                # 存储结果
                key = f"prompt-{i} persona-{j}"
                all_conversations[key].append(conversation_data)
                
                # 避免速率限制
                time.sleep(2)
        
        # 保存结果
        output_file = f"{SIMULATION_PATH}/m0/{SPLIT_NAME}.json"
        with open(output_file, "w") as f:
            json.dump(dict(all_conversations), f, indent=2)
            
        logging.info(f"Saved conversations to {output_file}")
        
        # 创建轮次文件
        for turn in range(1, MAX_TURNS + 1):
            turn_conversations = defaultdict(list)
            for key, convs in all_conversations.items():
                turn_conversations[key] = [
                    {
                        "task": conv["task"],
                        "persona": conv["persona"],
                        "conversation": conv["conversation"][:turn*2],  # 每轮包含一个问题和一个回答
                        "final_response": conv["final_response"] if turn == MAX_TURNS else ""
                    }
                    for conv in convs
                ]
            
            turn_file = f"{SIMULATION_PATH}/m0/{SPLIT_NAME}_turn-{turn}.json"
            with open(turn_file, "w") as f:
                json.dump(dict(turn_conversations), f, indent=2)
            
            logging.info(f"Saved turn {turn} conversations to {turn_file}")
        
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 处理命令行参数
    import sys
    for arg in sys.argv[1:]:
        if arg.startswith('--split=') or arg.startswith('split='):
            SPLIT_NAME = arg.split('=')[1]
    
    random.seed(1)  # 设置随机种子，确保结果可重现
    logging.info(f"Starting conversation simulation for split: {SPLIT_NAME}")
    main()