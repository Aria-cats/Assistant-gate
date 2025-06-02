import json
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# 路径配置
VERSION_2_BSFT = "v2-bsft"
BASE_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate"
SIMULATION_PATH = f"{BASE_PATH}/simulate-conversations/outputs/{VERSION_2_BSFT}"
FILTERED_PATH = f"{BASE_PATH}/log-probs/outputs/{VERSION_2_BSFT}/qa/m0"
PERSONAS_PATH = f"{BASE_PATH}/persona-generation/outputs/{VERSION_2_BSFT}"
MODELRESPONSE_PATH = f"{BASE_PATH}/sft/preprocess/outputs/{VERSION_2_BSFT}"
SFT_DATA_PATH = f"{BASE_PATH}/sft/train/data/{VERSION_2_BSFT}"

def extract_history(conversation):
    """提取对话历史"""
    if isinstance(conversation, dict):
        if "conversation" in conversation:
            return conversation["conversation"]
        if "messages" in conversation:
            return "\n".join([msg.get("content", "") for msg in conversation["messages"]])
    return conversation

def create_turns(conversation):
    """将对话分割为轮次"""
    if isinstance(conversation, str):
        # 根据您的对话格式分割对话轮次
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
    
    logging.info(f"加载筛选后的对话和模型回复...")
    
    # 加载筛选后的对话
    with open(f"{FILTERED_PATH}/{split_name}.json", 'r') as f:
        conversations = json.load(f)
    
    # 加载模型回复
    with open(f"{MODELRESPONSE_PATH}/m0/{split_name}.json", 'r') as f:
        model_responses = json.load(f)
    
    # 加载角色名称
    try:
        with open(f'{PERSONAS_PATH}/{split_name}_NAMES.json', "r") as f:
            names = json.load(f)
    except:
        logging.warning("未找到角色名称数据，使用默认值")
        names = [f"用户{i+1}" for i in range(10)]  # 假设最多10个角色
    
    # 准备训练数据
    targets = []
    
    for key, conv_list in conversations.items():
        logging.info(f"处理 {key}...")
        
        if key not in model_responses:
            logging.warning(f"未找到 {key} 的模型回复，跳过")
            continue
        
        bsft_responses = model_responses[key]
        
        # 提取角色索引
        p_idx = int(key[key.find('persona-') + len('persona-'):].strip())
        
        for c_idx, conversation in enumerate(conv_list):
            if c_idx >= len(bsft_responses):
                logging.warning(f"模型回复数量不足，跳过 {key} 中的对话 {c_idx}")
                continue
                
            # 提取对话历史
            history = extract_history(conversation)
            turns = create_turns(history)
            
            # 添加角色名称和原始任务
            if len(turns) > 0:
                turns[0] = f"My name is {names[p_idx] if p_idx < len(names) else 'User'}.\n\n{turns[0].strip()}"
            
            # 添加原始任务以提示回复
            if len(turns) > 0:
                turns[-1] = f"{turns[-1]}\n\n{turns[0].strip()}"
            
            # 添加模型回复
            turns.append(bsft_responses[c_idx])
            
            # 创建消息格式
            messages = []
            for i, turn in enumerate(turns):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": turn})
            
            targets.append(messages)
    
    # 创建输出目录
    os.makedirs(f"{SFT_DATA_PATH}/m0", exist_ok=True)
    
    # 保存训练数据
    output_path = f"{SFT_DATA_PATH}/m0/{split_name}.json"
    with open(output_path, 'w') as f:
        json.dump(targets, f, indent=2)
    
    logging.info(f"已保存训练数据到 {output_path}")

if __name__ == "__main__":
    main()