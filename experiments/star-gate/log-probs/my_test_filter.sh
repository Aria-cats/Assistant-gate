#!/bin/bash

echo "===== 计算测试拆分的对数概率 ====="
cd ~/assistant-gate/experiments/star-gate/log-probs
mkdir -p outputs/v2-bsft/qa/m0
python my_calculate_log_probs.py --split=test
echo "===== 计算完成 ====="