import json
import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置日志
logging.basicConfig(level=logging.INFO)

# 路径配置
VERSION_2_BSFT = "v2-bsft"
BASE_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate"
SIMULATION_PATH = f"{BASE_PATH}/simulate-conversations/outputs/{VERSION_2_BSFT}"
FILTERED_PATH = f"{BASE_PATH}/log-probs/outputs/{VERSION_2_BSFT}/filtered/m0"
PERSONAS_PATH = f"{BASE_PATH}/persona-generation/outputs/{VERSION_2_BSFT}"
MODELRESPONSE_PATH = f"{BASE_PATH}/sft/preprocess/outputs/{VERSION_2_BSFT}"

def extract_history(conversation):
    """提取对话历史"""
    # 根据您的对话格式进行调整
    if isinstance(conversation, dict):
        if "conversation" in conversation:
            return conversation["conversation"]
        if "messages" in conversation:
            return "\n".join([msg.get("content", "") for msg in conversation["messages"]])
    return conversation

def create_turns(conversation):
    """将对话分割为轮次"""
    # 简化实现，根据您的格式可能需要调整
    if isinstance(conversation, str):
        # 根据您的对话格式分割对话轮次
        # 示例: 以"用户:"和"助手:"分割
        turns = []
        parts = conversation.split("Human: ")
        for part in parts[1:]:  # 跳过第一个空部分
            if "Assistant: " in part:
                user_msg, asst_part = part.split("Assistant: ", 1)
                turns.append(user_msg.strip())
                turns.append(asst_part.strip())
            else:
                turns.append(part.strip())
        return turns
    return [conversation]

def main():
    # 设置运行参数
    split_name = "test"  # 可以设置为 A, B 或 test
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # 使用您熟悉的模型
    k = 1  # top-k 对话数
    
    logging.info(f"加载模型 {model_name}...")
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # 加载筛选后的对话
    logging.info(f"加载筛选后的对话...")
    filtered_path = f"{FILTERED_PATH}/{split_name}.json"
    with open(filtered_path, 'r') as f:
        conversations = json.load(f)
    
    # 加载角色
    logging.info(f"加载角色数据...")
    try:
        with open(f'{PERSONAS_PATH}/{split_name}.json', "r") as f:
            personas = json.load(f)
        with open(f'{PERSONAS_PATH}/{split_name}_NAMES.json', "r") as f:
            names = json.load(f)
    except:
        logging.warning("未找到角色数据，使用默认值")
        personas = ["一个普通用户" for _ in range(3)]  # 假设有3个角色
        names = [f"用户{i+1}" for i in range(3)]
    
    # 生成回复
    output_responses = {}
    
    for j, persona in enumerate(personas[:2]):  # 只处理前两个角色以节约时间
        logging.info(f"开始生成角色 {j} 的回复...")
        
        # 找到属于这个角色的对话
        persona_keys = [key for key in conversations.keys() if key.strip().endswith(f'persona-{j}')]
        
        for key in persona_keys:
            output_responses[key] = []
            
            for conversation in conversations[key]:
                # 提取对话历史
                history = extract_history(conversation)
                turns = create_turns(history)
                
                # 创建提示
                prompt = f"My name is {names[j].strip()}.\n\n{turns[0].strip()}\n\n"
                if len(turns) > 1:
                    for i in range(1, len(turns)):
                        role = "Human" if i % 2 == 0 else "Assistant"
                        prompt += f"{role}: {turns[i]}\n\n"
                
                # 再次添加原始任务以提示模型生成回答
                prompt += f"{turns[0].strip()}"
                
                # 生成回复
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
                
                # 解码并添加到输出
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                output_responses[key].append(response)
    
    # 创建输出目录
    os.makedirs(f"{MODELRESPONSE_PATH}/m0", exist_ok=True)
    
    # 保存生成的回复
    output_path = f"{MODELRESPONSE_PATH}/m0/{split_name}.json"
    with open(output_path, 'w') as f:
        json.dump(output_responses, f, indent=2)
    
    logging.info(f"已保存模型回复到 {output_path}")

if __name__ == "__main__":
    main()