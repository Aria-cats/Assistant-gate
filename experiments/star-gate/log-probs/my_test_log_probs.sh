#!/bin/bash

echo "===== 计算测试拆分的对数概率 ====="
cd ~/assistant-gate/experiments/star-gate/log-probs
mkdir -p outputs/v2-bsft/qa/m0
# 不传递任何参数，直接运行Python脚本
python my_calculate_log_probs.py
echo "===== 计算完成 ====="
chmod +x my_test_log_probs.sh