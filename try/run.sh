#!/bin/bash

# 三维高斯泼溅重建系统运行脚本

# 设置环境
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 1. 数据预处理
echo "步骤1: 数据预处理"
python data_preprocess.py \
    --data_dir "raw_data/bicycle" \
    --output_dir "processed_data/bicycle"

# 2. 训练模型
echo "步骤2: 训练模型"
python train.py \
    --data_dir "processed_data/bicycle" \
    --output_dir "outputs/bicycle" \
    --epochs 1000 \
    --max_gaussians 100000 \
    --device cuda \
    --use_wandb

# 3. 评估模型
echo "步骤3: 评估模型"
python evaluate.py \
    --pred_dir "outputs/bicycle/renders" \
    --gt_dir "processed_data/bicycle/images" \
    --device cuda

# 4. 导出结果
echo "步骤4: 导出结果"
python export_results.py \
    --model_path "outputs/bicycle/checkpoint_epoch_1000.pth" \
    --output_dir "results/bicycle"