import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import argparse
from typing import BinaryIO

# (*** 新增: 导入 tqdm ***)
from tqdm import tqdm

try:
    from tokenizers import Tokenizer
except ImportError:
    print("="*50)
    print("错误: 缺少 'tokenizers' 库。")
    print("请立即运行: pip install tokenizers")
    print("="*50)
    exit(1)


# 从您提供的文件中导入必要的模块
from LM import LM
from utils import AdamW, save_checkpoint, load_checkpoint

# ==============================================================================
# 数据加载器 (Data Loader)
# ==============================================================================

def get_batch(input_data, label_data, batch_size, context_length, device):
    """
    (*** 变化点: 恢复为原始的随机抓取逻辑 ***)
    从 .npy 文件中随机获取一批输入和标签
    """
    # 随机选择 batch_size 个索引
    ix = torch.randint(len(input_data), (batch_size,))
    
    # 从 .npy 文件中提取数据
    # 注意: .npy 数据是 (N, T) 形状
    x = torch.stack([torch.from_numpy(input_data[i].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(label_data[i].astype(np.int64)) for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

# ==============================================================================
# 评估函数 (Evaluation Function)
# ==============================================================================

@torch.no_grad()
def evaluate(model, val_inputs, val_labels, batch_size, context_length, device, eval_iters=100):
    """在验证集上评估模型性能。"""
    model.eval()
    losses = torch.zeros(eval_iters)
    # (*** 变化点: 评估循环使用 tqdm ***)
    pbar_eval = tqdm(range(eval_iters), desc="[评估中]", leave=False)
    for k in pbar_eval:
        X, Y = get_batch(val_inputs, val_labels, batch_size, context_length, device)
        
        logits = model(X)
        
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T), ignore_index=-100)
        
        losses[k] = loss.item()
        
    pbar_eval.close() # 关闭评估进度条
    model.train()
    return losses.mean()

# ==============================================================================
# 主训练函数 (Main Training Function)
# ==============================================================================

def train(args):
    # --- 1. 设置 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # --- 2. 加载分词器并获取 Vocab Size ---
    print(f"正在从 {args.tokenizer_file} 加载分词器...")
    try:
        tokenizer = Tokenizer.from_file(args.tokenizer_file)
        VOCAB_SIZE = tokenizer.get_vocab_size()
        print(f"分词器加载成功。词表大小: {VOCAB_SIZE}")
    except Exception as e:
        print(f"错误: 无法加载分词器文件 '{args.tokenizer_file}'. {e}")
        return

    # --- 3. 加载数据 ---
    print("正在加载数据 (内存映射)...")
    try:
        train_inputs = np.load(args.input_path, mmap_mode='r')
        train_labels = np.load(args.label_path, mmap_mode='r')
        val_inputs = np.load(args.val_input_path, mmap_mode='r')
        val_labels = np.load(args.val_label_path, mmap_mode='r')
        print(f"数据加载完成。训练集序列数: {len(train_inputs)}, 验证集序列数: {len(val_inputs)}")
    except FileNotFoundError:
        print("错误: 找不到 .npy 数据文件。")
        print("请确保您已经成功运行了 'prepare_data.py' 脚本。")
        return

    # --- 4. 初始化模型和优化器 ---
    print("正在初始化模型...")
    model = LM(
        vocab_size=VOCAB_SIZE, 
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32 if device == 'cuda' else torch.float32 
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # --- 5. 从检查点恢复 (可选) ---
    start_iter = 0
    if args.resume_from:
        print(f"正在从 {args.resume_from} 恢复...")
        # (*** 变化点: 恢复步数 ***)
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    # --- 6. 训练循环 (*** 变化点: 基于 Step ***) ---
    print(f"开始训练，设备: {device}")
    model.train()
    
    # (*** 变化点: 早停变量 ***)
    best_val_loss = float('inf')
    patience_counter = 0
    
    # (*** 变化点: 主循环使用 tqdm，基于步数 ***)
    pbar = tqdm(range(start_iter, args.max_steps), desc="[训练中]", leave=True)
    
    for i in pbar:
        # 获取一批数据
        inputs, targets = get_batch(train_inputs, train_labels, args.batch_size, args.context_length, device)
        
        # 前向传播和损失计算
        logits = model(inputs)
        
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index=-100)
        
        # 反向传播和参数更新
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # (*** 变化点: 在 tqdm 进度条上显示损失 ***)
        pbar.set_postfix({'批次损失': f'{loss.item():.4f}'})

        # --- 7. 日志、评估和检查点 ---
        
        # (*** 变化点: 移除 log_interval，评估逻辑移至 eval_interval ***)

        if (i + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_inputs, val_labels, args.batch_size, args.context_length, device)
            
            # (*** 变化点: 更新 tqdm 描述 ***)
            pbar.write("-" * 50)
            pbar.write(f"迭代 {i+1} | 训练损失 (当前批次): {loss.item():.4f} | 验证损失: {val_loss:.4f}")
            pbar.write("-" * 50)

            # (*** 变化点: 早停逻辑 ***)
            if args.early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                    pbar.write(f"*** 新的最佳验证损失: {best_val_loss:.4f}。正在保存最佳模型... ***")
                    save_checkpoint(model, optimizer, i + 1, best_model_path)
                else:
                    patience_counter += 1
                    pbar.write(f"*** 验证损失未改善。耐心计数: {patience_counter}/{args.early_stopping_patience} ***")

                if patience_counter >= args.early_stopping_patience:
                    pbar.write(f"*** 早停触发: 验证损失在 {args.early_stopping_patience} 次评估中未改善。 ***")
                    pbar.write(f"*** 停止训练于迭代 {i+1} ***")
                    break # 跳出训练循环

        if (i + 1) % args.checkpoint_interval == 0:
            # 常规检查点保存
            checkpoint_path = os.path.join(args.checkpoint_dir, f'ckpt_iter_{i+1}.pth')
            pbar.write(f"正在保存常规检查点: {checkpoint_path}")
            save_checkpoint(model, optimizer, i + 1, checkpoint_path)
            
    pbar.close() # 确保训练循环结束后关闭
    print("训练完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer 语言模型训练脚本 (按 Step)')
    
    # 数据和路径参数
    parser.add_argument('--input_path', type=str, default='./data/train_inputs.npy', help='训练输入 .npy 文件路径')
    parser.add_argument('--label_path', type=str, default='./data/train_labels.npy', help='训练标签 .npy 文件路径')
    parser.add_argument('--val_input_path', type=str, default='./data/val_inputs.npy', help='验证输入 .npy 文件路径')
    parser.add_argument('--val_label_path', type=str, default='./data/val_labels.npy', help='验证标签 .npy 文件路径')
    
    parser.add_argument('--tokenizer_file', type=str, default='./tokenizer/tokenizer.json', help='分词器 .json 文件路径')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='检查点保存目录')
    parser.add_argument('--resume_from', type=str, default=None, help='从指定检查点恢复训练')
    
    # 模型超参数
    parser.add_argument('--context_length', type=int, default=2048, help='上下文长度')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 层数')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=384, help='前馈网络中间层维度')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE 的 theta 参数')

    # (*** 变化点: 恢复为 Step-based 参数 ***)
    # 优化器和训练超参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_steps', type=int, default=1000, help='最大训练步数 (例如 50000)')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪值 (0 表示不裁剪)')

    # (*** 变化点: 恢复为 Step-based 参数 ***)
    # 日志和保存参数
    # parser.add_argument('--log_interval', type=int, default=10, help='(已移除) 记录训练日志的间隔步数')
    parser.add_argument('--eval_interval', type=int, default=250, help='运行评估的间隔步数 (例如 250)')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='保存检查点的间隔步数 (例如 1000)')
    
    # (*** 变化点: 恢复为 Step-based 参数 ***)
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='早停的耐心值 (单位: 评估次数，设为 0 禁用)')
    
    args = parser.parse_args()
    
    train(args)