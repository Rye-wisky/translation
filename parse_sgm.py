import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re # <-- 导入 regex

def parse_sgm_to_txt_regex(sgm_file_path: str, txt_file_path: str):
    """
    (备用方案) 基于 Regex 的回退解析器。
    """
    print(f"警告：XML 解析失败，尝试使用正则表达式... ({sgm_file_path})")
    # Regex 查找 <seg ...> ... </seg> 之间的文本
    # 这是一个简单的 Regex，适用于 AI Challenger 的格式
    seg_regex = re.compile(r"<seg[^>]*>(.*?)</seg>", re.DOTALL)
    
    line_count = 0
    try:
        with open(sgm_file_path, 'r', encoding='utf-8') as f_in:
            content = f_in.read()
        
        with open(txt_file_path, 'w', encoding='utf-8') as f_out:
            matches = seg_regex.finditer(content)
            for match in tqdm(matches, desc="Regex 提取"):
                text = match.group(1)
                
                # 手动解码基础的 XML 实体
                text = text.replace('&lt;', '<')
                text = text.replace('&gt;', '>')
                text = text.replace('&amp;', '&')
                text = text.replace('&quot;', '"')
                text = text.replace('&apos;', "'")
                text = text.replace('&ref;', '') # 保留
                
                # (重要) 移除其他可能残留的标签，比如 <p1.3/>
                text = re.sub(r"<[^>]+>", "", text)
                
                f_out.write(text.strip() + '\n')
                line_count += 1
        
        if line_count > 0:
            print(f"成功！已从 {sgm_file_path} 提取 {line_count} 行并保存至 {txt_file_path}")
        else:
            print(f"警告：Regex 未在 {sgm_file_path} 中找到任何 <seg>...</seg> 匹配项。")

    except Exception as e:
        print(f"Regex 备用方案失败: {e}")


def parse_sgm_to_txt(sgm_file_path: str, txt_file_path: str):
    """
    解析 SGM 文件，提取 <seg> 标签中的文本，并保存到 TXT 文件。
    """
    print(f"正在解析 {sgm_file_path}...")
    
    # 确保 SGM 文件存在
    if not os.path.exists(sgm_file_path):
        print(f"错误：找不到文件 {sgm_file_path}")
        print("请确保您已将 'valid.en-zh.en.sgm' 和 'valid.en-zh.zh.sgm' 放入 './data/' 文件夹。")
        return

    root = None
    try:
        # 尝试直接解析
        tree = ET.parse(sgm_file_path)
        root = tree.getroot()
    except ET.ParseError:
        print("XML解析失败。尝试清理并添加根标签...")
        try:
            with open(sgm_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # (修改) 手动处理常见的、可能导致失败的实体
            content = content.replace('&ref;', '') 
            content = content.replace('&apos;', "'")
            content = content.replace('&quot;', '"')

            xml_content = f"<root>{content}</root>"
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"错误：XML 解析器无法处理该文件: {e}")
            # (修改) 如果 XML 解析彻底失败，切换到 Regex 备用方案
            parse_sgm_to_txt_regex(sgm_file_path, txt_file_path)
            return

    # 提取所有 <seg> 标签的文本
    if root is not None:
        line_count = 0
        try:
            with open(txt_file_path, 'w', encoding='utf-8') as f_out:
                # (*** 修正点 ***)
                # 使用 .//seg 来查找所有 <seg> 标签
                for seg in tqdm(root.findall('.//seg'), desc="提取句子"):
                    
                    # 使用 .itertext() 来获取所有内部文本，
                    # 无论是否有 <p1.3/> 这样的子标签。
                    all_text = "".join(seg.itertext())
                    
                    if all_text:
                        # 清理可能残留的实体引用
                        all_text = all_text.replace('&ref;', '')
                        all_text = all_text.replace('&apos;', "'")
                        all_text = all_text.replace('&quot;', '"')
                        f_out.write(all_text.strip() + '\n')
                        line_count += 1
                    else:
                        # 处理空标签
                        f_out.write('\n')
            
            print(f"成功！已从 {sgm_file_path} 提取 {line_count} 行并保存至 {txt_file_path}")
        except Exception as e:
            print(f"在写入 {txt_file_path} 时发生错误: {e}")


if __name__ == "__main__":
    DATA_DIR = "./data/"
    
    # 定义输入和输出文件路径
    files_to_parse = [
        {
            "input": os.path.join(DATA_DIR, "valid.en-zh.en.sgm"),
            "output": os.path.join(DATA_DIR, "val.en")
        },
        {
            "input": os.path.join(DATA_DIR, "valid.en-zh.zh.sgm"),
            "output": os.path.join(DATA_DIR, "val.zh")
        }
    ]
    
    # 确保 data 目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("--- 开始解析 SGM 验证文件 ---")
    
    for pair in files_to_parse:
        # (修改) 确保在重新运行前删除旧的输出文件
        if os.path.exists(pair["output"]):
            os.remove(pair["output"])
            print(f"已删除旧文件: {pair['output']}")
            
        parse_sgm_to_txt(pair["input"], pair["output"])
    
    print("\n--- 解析完成 ---")
    print(f"现在您在 '{DATA_DIR}' 目录中应该有：")
    print("  val.en (纯文本)")
    print("  val.zh (纯文本)")
    print("\n您现在可以运行 'prepare_data.py' 脚本了。")

