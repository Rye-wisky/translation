import torch
import torch.nn.functional as F
import argparse
import json
import os

from LM import LM # 从您的 LM.PY 文件导入模型

# ==============================================================================
# 简易解码器 (Simple Decoder)
# ==============================================================================

class SimpleDecoder:
    """
    一个简化的解码器，用于将 token ID 转换回文本。
    它通过加载一个词汇表文件来实现。
    """
    def __init__(self, vocab_path):
        # 假设词汇表是一个 JSON 文件，将 ID (字符串形式) 映射到其字节表示
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                # 在BPE训练中，词汇表通常是 {int: bytes}
                # 如果保存为JSON，整数键会变成字符串，所以需要转换回来
                vocab_str_keys = json.load(f)
            self.id_to_token = {int(k): v.encode('latin-1') for k, v in vocab_str_keys.items()}
            print(f"成功加载包含 {len(self.id_to_token)} 个条目的词汇表。")
        except Exception as e:
            print(f"加载词汇表时出错: {e}")
            self.id_to_token = {i: bytes([i]) for i in range(256)} # 后备方案：使用原始字节
            print("回退到简单的字节-ID映射 (0-255)。")

    def decode(self, token_ids):
        """将 token ID 列表解码为字符串。"""
        byte_chunks = [self.id_to_token.get(token_id, b'') for token_id in token_ids]
        all_bytes = b''.join(byte_chunks)
        return all_bytes.decode('utf-8', errors='replace')

# ==============================================================================
# 核心生成函数 (Core Generation Function)
# ==============================================================================

@torch.no_grad()
def generate(model,
             prompt_tokens,
             max_new_tokens,
             temperature=1.0,
             top_p=0.9):
    """
    根据给定的提示，从模型生成一系列 token。

    参数:
        model: 训练好的 Transformer 模型。
        prompt_tokens: 提示的 token ID 张量，形状为 (1, T)。
        max_new_tokens: 要生成的最大新 token 数量。
        temperature: 温度缩放参数。值越高，输出越随机。
        top_p: 核心采样参数。过滤掉累积概率低于该值的 token。
    
    返回:
        torch.Tensor: 包含提示和生成 token 的完整序列。
    """
    model.eval()
    
    idx = prompt_tokens

    for _ in range(max_new_tokens):
        # 1. 如果上下文过长，则进行裁剪以匹配模型的 context_length
        context_length = model.context_length
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]
        
        # 2. 从模型获取 logits
        logits = model(idx_cond)
        
        # 3. 只关注最后一个时间步的 logits
        logits = logits[:, -1, :] # 形状: (B, vocab_size)
        
        # 4. 应用温度缩放
        if temperature > 0:
            logits = logits / temperature
        
        # 5. 应用 Top-p (核心) 采样
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float('-inf')

        # 6. 从修改后的 logits 中采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 7. 将采样到的 token 添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# ==============================================================================
# 脚本主入口
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从训练好的 Transformer 模型生成文本。')
    
    parser.add_argument('--checkpoint_path', type=str, required=True, help='模型检查点文件的路径 (.pth)。')
    # BPE 训练会生成一个词汇表文件，这里需要用到它来解码
    parser.add_argument('--vocab_path', type=str, required=True, help='词汇表文件的路径 (例如 .json)。')
    
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='用于开始生成的提示文本。')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='要生成的最大新 token 数量。')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度，越高越随机。')
    parser.add_argument('--top_p', type=float, default=0.9, help='核心采样阈值，0 表示禁用。')
    
    # 再次定义模型参数，以便我们可以重新构建模型结构来加载权重
    parser.add_argument('--vocab_size', type=int, default=10000, help='词汇表示量大小')
    parser.add_argument('--context_length', type=int, default=256, help='上下文长度')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 层数')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=1344, help='前馈网络中间层维度')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE 的 theta 参数')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用的设备: {device}")
    
    # --- 1. 从检查点加载模型 ---
    # 注意：为了加载 state_dict，我们需要先用与训练时完全相同的超参数来实例化模型。
    # 最佳实践是将这些超参数保存在检查点中。
    try:
        model = LM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=device
        ).to(device)
        
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("模型已从检查点成功加载。")
    except Exception as e:
        print(f"从检查点加载模型时出错: {e}")
        exit(1)

    # --- 2. 设置解码器 ---
    decoder = SimpleDecoder(args.vocab_path)
    # 这是一个简化的编码器，对于一个完整的BPE分词器，您需要使用训练时生成的合并规则。
    prompt_encoded = list(args.prompt.encode('utf-8'))
    prompt_tokens = torch.tensor(prompt_encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    # --- 3. 生成文本 ---
    print(f"\n--- 正在从提示生成文本 ---\n'{args.prompt}'\n" + "-"*50)
    
    generated_tokens_tensor = generate(
        model,
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    generated_tokens_list = generated_tokens_tensor[0].tolist()
    
    # --- 4. 解码并打印结果 ---
    generated_text = decoder.decode(generated_tokens_list)
    print("\n--- 生成的输出 ---\n")
    print(generated_text)
    print("\n" + "-"*50)
