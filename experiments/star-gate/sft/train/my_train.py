import os
import logging
import random
import json
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# 配置日志
logging.basicConfig(level=logging.INFO)

# 路径配置
VERSION_2_BSFT = "v2-bsft"
BASE_PATH = f"/hpc2hdd/home/yliu433/assistant-gate/experiments/star-gate"
SFT_DATA_PATH = f"{BASE_PATH}/sft/train/data/{VERSION_2_BSFT}"
OUTPUT_PATH = f"{BASE_PATH}/sft/train/outputs/{VERSION_2_BSFT}"

class DataCollatorForSupervisedDataset(DataCollatorForLanguageModeling):
    """用于监督数据集的数据整理器"""
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)
        
    def __call__(self, examples):
        # 简化版本，将所有示例连接为单个文本
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for example in examples:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(input_ids)
        
        # 填充到最大长度
        batch = self.tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )
        
        return batch

def preprocess(targets, tokenizer):
    """预处理训练数据"""
    
    def tokenize_function(examples):
        # 连接消息为字符串
        texts = []
        for message_list in examples["messages"]:
            text = tokenizer.apply_chat_template(message_list)
            texts.append(text)
        
        # 分词
        result = tokenizer(texts, truncation=True, max_length=2048)
        return result
    
    # 创建数据集
    messages_data = {"messages": targets}
    dataset = Dataset.from_dict(messages_data)
    
    # 应用分词
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["messages"]
    )
    
    return dataset

def main():
    # 设置随机种子
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    
    # 设置运行参数
    split_name = "test"  # 可以设置为 A, B 或 test
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # 使用您熟悉的模型
    validation_split_size = 0.1  # 验证集大小
    
    logging.info(f"在 {split_name} 拆分上训练模型...")
    
    # 加载模型和分词器
    logging.info(f"加载模型 {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 加载训练数据
    logging.info(f"加载训练数据...")
    data_path = f"{SFT_DATA_PATH}/m0/{split_name}.json"
    with open(data_path, 'r') as f:
        targets = json.load(f)
    
    # 预处理数据
    dataset = preprocess(targets=targets, tokenizer=tokenizer)
    dataset = dataset.shuffle(seed=42)
    
    # 拆分训练集和验证集
    split_dataset = dataset.train_test_split(test_size=validation_split_size)
    
    # 设置训练参数
    output_dir = f"{OUTPUT_PATH}/m0/{split_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/final", exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # 根据您的GPU内存调整
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # 累积梯度以模拟更大的批次
        num_train_epochs=1,             # 减少训练轮次以加快训练
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,  # 使用混合精度训练
    )
    
    # 创建数据整理器
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    logging.info("开始训练...")
    trainer.train()
    
    # 保存模型
    logging.info("保存模型...")
    trainer.save_model(output_dir=f"{output_dir}/final")
    
    logging.info("训练完成！")

if __name__ == "__main__":
    main()