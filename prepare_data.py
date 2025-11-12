import os
import json
import numpy as np
from tqdm import tqdm
import multiprocessing # <-- 1. 导入
from typing import Optional, Tuple

# 导入高效的 tokenizers 库
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
except ImportError:
    print("="*50)
    print("错误: 缺少 'tokenizers' 库。")
    print("请立即运行: pip install tokenizers")
    print("="*50)
    exit(1)


# --- 配置 ---
DATA_DIR = "./data/"
TOKENIZER_DIR = "./tokenizer/"
TOKENIZER_FILE = os.path.join(TOKENIZER_DIR, "tokenizer.json")

VOCAB_SIZE = 32000
CONTEXT_LENGTH = 256 
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]

# ==============================================================================
# --- 多进程工作函数 (Worker Functions) ---
# 这些变量将在每个工作进程中被初始化
# ==============================================================================

G_TOKENIZER: Optional[Tokenizer] = None
G_MAX_LEN = 0
G_BOS_ID, G_EOS_ID, G_SEP_ID, G_PAD_ID = 0, 0, 0, 0

def init_worker(tokenizer_path: str, max_len: int):
    """
    初始化函数，在每个工作进程池启动时被调用。
    这避免了在主进程和子进程之间传递大型 tokenizer 对象。
    """
    global G_TOKENIZER, G_MAX_LEN, G_BOS_ID, G_EOS_ID, G_SEP_ID, G_PAD_ID
    
    print(f"[PID: {os.getpid()}] 正在加载分词器...")
    G_TOKENIZER = Tokenizer.from_file(tokenizer_path)
    G_MAX_LEN = max_len
    
    G_BOS_ID = G_TOKENIZER.token_to_id("[BOS]")
    G_EOS_ID = G_TOKENIZER.token_to_id("[EOS]")
    G_SEP_ID = G_TOKENIZER.token_to_id("[SEP]")
    G_PAD_ID = G_TOKENIZER.token_to_id("[PAD]")

    if None in [G_BOS_ID, G_EOS_ID, G_SEP_ID, G_PAD_ID]:
        print("错误：工作进程未能正确加载特殊 token。")
        # 在多进程中，抛出异常比 exit(1) 更好
        raise ValueError("无法在工作进程中加载特殊 token ID")

def _process_line_pair(line_pair: Tuple[str, str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    实际的工作函数，在工作进程中被调用。
    处理*单对*中英文行。
    """
    global G_TOKENIZER, G_MAX_LEN, G_BOS_ID, G_EOS_ID, G_SEP_ID, G_PAD_ID
    
    # G_TOKENIZER 应该已经在 init_worker 中被设置
    if G_TOKENIZER is None:
        return None

    zh_line, en_line = line_pair
    
    zh_line = zh_line.strip()
    en_line = en_line.strip()

    if not zh_line or not en_line:
        return None

    # 1. 构建提示（Prompt）
    prompt_text = f"翻译成英文: {zh_line}"
    prompt_ids = G_TOKENIZER.encode(prompt_text).ids
    
    # 2. 构建答案（Answer）
    answer_ids = G_TOKENIZER.encode(en_line).ids

    # 3. 组合输入序列
    input_ids = [G_BOS_ID] + prompt_ids + [G_SEP_ID] + answer_ids + [G_EOS_ID]
    
    # 4. 构建标签序列（应用损失掩码）
    prompt_mask = [-100] * (len(prompt_ids) + 2) # 覆盖 [BOS], prompt, [SEP]
    answer_mask = answer_ids + [G_EOS_ID]
    label_ids = prompt_mask + answer_mask

    # 5. 填充或截断
    if len(input_ids) > G_MAX_LEN:
        input_ids = input_ids[:G_MAX_LEN]
        label_ids = label_ids[:G_MAX_LEN]
    elif len(input_ids) < G_MAX_LEN:
        pad_count = G_MAX_LEN - len(input_ids)
        input_ids.extend([G_PAD_ID] * pad_count)
        label_ids.extend([-100] * pad_count)
    
    # 返回 NumPy 数组以优化内存
    return (
        np.array(input_ids, dtype=np.uint16), 
        np.array(label_ids, dtype=np.int32)
    )

# ==============================================================================
# --- 重构后的 `process_file_pair` (现在是多进程协调器) ---
# ==============================================================================

def process_file_pair(zh_path: str, en_path: str, tokenizer_file: str, max_len: int):
    """
    使用多进程并行处理一对中英文文件。
    """
    print(f"正在处理: {zh_path} 和 {en_path}")
    
    if not os.path.exists(zh_path) or not os.path.exists(en_path):
        print(f"警告: 找不到文件对 {zh_path} 或 {en_path}。跳过。")
        return [], []

    # 1. (快速) 统计总行数，用于 tqdm 进度条
    print("正在统计总行数...")
    with open(zh_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    all_inputs = []
    all_labels = []

    # 2. 设置进程池
    num_cores = multiprocessing.cpu_count()
    print(f"开始使用 {num_cores} 个核心并行处理...")

    # 3. (重要) 不使用 .readlines()，而是懒加载
    with open(zh_path, 'r', encoding='utf-8') as f_zh, \
         open(en_path, 'r', encoding='utf-8') as f_en:
        
        # 创建一个行对的 *生成器* (generator)
        line_pairs = zip(f_zh, f_en)
        
        # 4. 创建进程池
        # initializer 在每个 worker 启动时调用
        # initargs 是传递给 initializer 的参数
        with multiprocessing.Pool(processes=num_cores, 
                                  initializer=init_worker, 
                                  initargs=(tokenizer_file, max_len)) as pool:
            
            # 5. imap_unordered:
            #   - imap: 按顺序返回结果 (但处理是并行的)
            #   - unordered: 哪个任务先完成，就先返回哪个 (更快)
            #   - chunksize: 一次性给一个 worker 分配 1024 个任务，
            #                这比逐个分配（默认 chunksize=1）高效得多。
            chunksize = 1024 
            results_iter = pool.imap_unordered(_process_line_pair, line_pairs, chunksize=chunksize)
            
            # 6. 收集结果
            for result in tqdm(results_iter, total=total_lines, desc="构建序列 (并行)"):
                if result is not None:
                    all_inputs.append(result[0])
                    all_labels.append(result[1])

    return all_inputs, all_labels


def main():
    # --- 步骤 1: (如果需要) 为BPE训练合并语料库 ---
    print(f"--- 步骤 1: 检查 BPE 训练语料库 ---")
    corpus_path = os.path.join(DATA_DIR, "corpus_for_bpe.txt")

    if not os.path.exists(corpus_path):
        print(f"正在创建 BPE 语料库: {corpus_path}")
        with open(corpus_path, 'w', encoding='utf-8') as f_out:
            for lang in ['zh', 'en']:
                for split in ['train', 'val']:
                    file_path = os.path.join(DATA_DIR, f"{split}.{lang}")
                    if os.path.exists(file_path):
                        print(f"正在添加: {file_path}")
                        with open(file_path, 'r', encoding='utf-8') as f_in:
                            for line in tqdm(f_in, desc=f"处理 {split}.{lang}"):
                                line = line.strip()
                                if line:
                                    f_out.write(line + "\n")
                    else:
                        print(f"警告: 未找到 {file_path}，将跳过。")
        print(f"BPE 语料库 {corpus_path} 创建完成。")
    else:
        print(f"BPE 语料库 {corpus_path} 已存在。")


    # --- 步骤 2: 训练 BPE 分词器 (使用 tokenizers 库) ---
    print(f"--- 步骤 2: 训练分词器 (使用 'tokenizers' 库) ---")
    
    if not os.path.exists(TOKENIZER_FILE):
        print(f"正在训练 BPE 分词器 (词表大小: {VOCAB_SIZE})...")
        os.makedirs(TOKENIZER_DIR, exist_ok=True)
        
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=SPECIAL_TOKENS
        )
        
        tokenizer.train([corpus_path], trainer)
        tokenizer.save(TOKENIZER_FILE)

        print(f"分词器文件已保存到 {TOKENIZER_FILE}")
    else:
        print(f"分词器文件 {TOKENIZER_FILE} 已存在。跳过训练。")

    
    # --- 步骤 3: 检查分词器加载 ---
    print(f"--- 步骤 3: 检查分词器 ---")
    try:
        # 只是为了检查，实际加载在 worker 中完成
        tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"分词器加载成功。实际词表大小: {actual_vocab_size}")
        del tokenizer # 释放内存
        
    except Exception as e:
        print(f"错误: 无法从 {TOKENIZER_FILE} 加载分词器文件。{e}")
        print("请删除 tokenizer 目录并重新运行脚本。")
        exit(1)


    # --- 步骤 4: 处理数据并应用掩码 ---
    print(f"--- 步骤 4: 处理训练集和验证集 ---")

    # 处理训练集
    train_inputs_path = os.path.join(DATA_DIR, "train_inputs.npy")
    train_labels_path = os.path.join(DATA_DIR, "train_labels.npy")
    if not os.path.exists(train_inputs_path) or not os.path.exists(train_labels_path):
        train_inputs, train_labels = process_file_pair(
            os.path.join(DATA_DIR, "train.zh"),
            os.path.join(DATA_DIR, "train.en"),
            TOKENIZER_FILE,     # <-- 传递路径，而不是对象
            CONTEXT_LENGTH
        )
        if train_inputs:
            print("Tokenization 完成。正在转换列表为 NumPy 数组并保存...")
            # (注意: 这一步仍然很慢且消耗内存，但 Tokenization 已经快很多了)
            np.save(train_inputs_path, np.array(train_inputs))
            np.save(train_labels_path, np.array(train_labels))
            print(f"训练集已保存。序列数量: {len(train_inputs)}")
    else:
        print("训练集 .npy 文件已存在。跳过处理。")

    # 处理验证集
    val_inputs_path = os.path.join(DATA_DIR, "val_inputs.npy")
    val_labels_path = os.path.join(DATA_DIR, "val_labels.npy")
    if not os.path.exists(val_inputs_path) or not os.path.exists(val_labels_path):
        val_inputs, val_labels = process_file_pair(
            os.path.join(DATA_DIR, "val.zh"),
            os.path.join(DATA_DIR, "val.en"),
            TOKENIZER_FILE,     # <-- 传递路径
            CONTEXT_LENGTH
        )
        if val_inputs:
            print("Tokenization 完成。正在转换列表为 NumPy 数组并保存...")
            np.save(val_inputs_path, np.array(val_inputs))
            np.save(val_labels_path, np.array(val_labels))
            print(f"验证集已保存。序列数量: {len(val_inputs)}")
    else:
        print("验证集 .npy 文件已存在。跳过处理。")

    print("\n" + "="*50)
    print("--- 数据准备完成 ---")
    print("您现在可以运行 train.py 来开始训练。")
    print(f"您的分词器位于: {TOKENIZER_FILE}")
    print(f"实际词表大小: {actual_vocab_size}")
    print("\n推荐的训练命令 (已移除 --vocab_size):")
    print(f"python train.py --tokenizer_file {TOKENIZER_FILE} --context_length {CONTEXT_LENGTH} ... (其他参数)")
    print("="*50)


if __name__ == "__main__":
    # 在使用 multiprocessing 时，将 main() 调用放在
    # 'if __name__ == "__main__":' 保护块中至关重要
    main()