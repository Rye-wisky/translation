import torch
import argparse
import sys
from LM import LM

# (*** 变化点: 导入与训练时一致的 tokenizers 库 ***)
try:
    from tokenizers import Tokenizer
except ImportError:
    print("="*50)
    print("错误: 缺少 'tokenizers' 库。")
    print("请立即运行: pip install tokenizers")
    print("="*50)
    sys.exit(1)

# (*** 变化点: 直接在此处定义常量 ***)
# 我们不再从 prepare_data.py 导入，以避免在导入时意外执行该脚本。
SPECIAL_TOKENS = {
    "BOS": "[BOS]", # 序列开始
    "EOS": "[EOS]", # 序列结束
    "SEP": "[SEP]", # 分隔符 (用于分隔源和目标)
    "PAD": "[PAD]", # 填充
}
PROMPT_TEMPLATE = "请将以下中文翻译成英文: "
# (*** 变化点结束 ***)


def load_model(checkpoint_path, device):
    """
    (*** 变化点: 自动从检查点加载模型参数 ***)
    加载模型和检查点。
    """
    
    # 1. 加载检查点
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"错误: 无法加载检查点文件 '{checkpoint_path}'. {e}")
        sys.exit(1)

    # 2. 提取模型参数 (model_args)
    if 'model_args' not in checkpoint:
        print(f"错误: 检查点 '{checkpoint_path}' 中缺少 'model_args'。")
        print("请确保您的训练脚本 (utils.py 中的 save_checkpoint) 正在保存 'model_args'。")
        print("请参考之前的说明来修改您的 LM.py 和 utils.py。")
        sys.exit(1)
        
    model_args = checkpoint['model_args']

    # 3. 初始化模型
    try:
        model = LM(
            **model_args,
            device=device 
        ).to(device)
    except Exception as e:
        print(f"错误: 使用 'model_args' 初始化 LM 失败。{e}")
        print(f"加载的参数: {model_args}")
        sys.exit(1)
    
    # 4. 加载模型状态
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"加载 model_state_dict 失败: {e}")
        print("提示: 确保模型定义 (LM.py) 与训练时完全一致。")
        sys.exit(1)

    model.eval()
    return model

@torch.no_grad()
def generate_translation(
    model: LM,
    tokenizer: Tokenizer,
    prompt_text: str,
    device: str,
    max_new_tokens=100
):
    """使用模型生成翻译"""
    
    try:
        bos_id = tokenizer.token_to_id(SPECIAL_TOKENS["BOS"])
        eos_id = tokenizer.token_to_id(SPECIAL_TOKENS["EOS"])
        sep_id = tokenizer.token_to_id(SPECIAL_TOKENS["SEP"])
    except Exception as e:
        print(f"错误: 无法在分词器中找到特殊 token ID。{e}")
        print("请确保您的 tokenizer.json 文件包含 [BOS], [EOS], [SEP] 等特殊 token。")
        return "[错误: 检查分词器]"

    # 1. 格式化并编码提示
    formatted_prompt = f"{SPECIAL_TOKENS['BOS']} {PROMPT_TEMPLATE}{prompt_text.strip()} {SPECIAL_TOKENS['SEP']}"
    
    try:
        input_ids = tokenizer.encode(formatted_prompt).ids
    except Exception as e:
        print(f"错误: 编码提示时出错: {e}")
        return "[错误: 编码失败]"
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated_ids = []

    # 2. 自回归生成
    # (*** 变化点: 增加导入 tqdm ***)
    from tqdm import tqdm
    pbar_gen = tqdm(range(max_new_tokens), desc="[生成中]", leave=False)
    
    for _ in pbar_gen:
        # (*** 变化点: 假设 LM 保存了 self.context_length ***)
        # (请确保在 LM.py 的 __init__ 中添加了 self.context_length = context_length)
        if not hasattr(model, 'context_length'):
             print("错误: 'model' 对象没有 'context_length' 属性。")
             print("请确保您已在 LM.py 的 __init__ 中添加 self.context_length = context_length。")
             return "[模型配置错误]"

        if input_tensor.shape[1] > model.context_length:
            input_tensor = input_tensor[:, -model.context_length:]

        logits = model(input_tensor)
        
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        
        if next_token_id.item() == eos_id:
            break
            
        generated_ids.append(next_token_id.item())
        input_tensor = torch.cat([input_tensor, next_token_id.unsqueeze(0)], dim=1)

    pbar_gen.close()

    # 6. 解码
    translation = tokenizer.decode(generated_ids)
    return translation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='中英翻译推理脚本')
    
    # 路径和设置
    parser.add_argument('--tokenizer_file', type=str, default='./tokenizer/tokenizer.json', help='分词器 .json 文件路径')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='训练好的模型检查点 (.pth 文件)')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='生成翻译的最大 token 数量')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载分词器
    print(f"正在从 {args.tokenizer_file} 加载分词器...")
    try:
        tokenizer = Tokenizer.from_file(args.tokenizer_file)
    except Exception as e:
        print(f"错误: 找不到或无法加载分词器文件 {args.tokenizer_file}. {e}")
        sys.exit(1)
        
    # 2. 加载模型
    print(f"正在从 {args.checkpoint_path} 加载模型...")
    model = load_model(args.checkpoint_path, device)
    print("模型加载成功。")

    # 3. 交互式翻译
    print(f"\n--- 中英翻译（设备: {device}）---")
    print("--- (输入 'q' 退出) ---")
    while True:
        try:
            prompt_text = input("请输入中文: ")
            if prompt_text.strip().lower() == 'q':
                break
            if not prompt_text.strip():
                continue

            translation = generate_translation(
                model, 
                tokenizer, 
                prompt_text, 
                device, 
                max_new_tokens=args.max_new_tokens
            )
            
            print(f"英文翻译: {translation}")
            print("-" * 20)

        except KeyboardInterrupt:
            print("\n退出。")
            break 