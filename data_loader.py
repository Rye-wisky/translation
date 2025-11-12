import torch
from torch import nn,Tensor
import numpy.typing as npt
import numpy as np

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从数据集中随机获取一批用于训练语言模型的数据。

    这个函数会从一个大的 token ID 序列中随机选择 `batch_size` 个起始点，
    然后为每个起始点提取一个长度为 `context_length` 的输入序列 (x) 和
    一个对应的目标序列 (y)。目标序列是输入序列向右移动一个位置。

    参数:
        data (np.ndarray): 包含 token ID 的一维 NumPy 数组。
        batch_size (int): 每个批次中的序列数量。
        context_length (int): 每个输入序列的长度。
        device (str): PyTorch 设备，例如 'cpu', 'cuda', 或 'mps'。

    返回:
        tuple[torch.Tensor, torch.Tensor]: 一个包含两个张量的元组 (x, y)。
                                           x 是输入张量，形状为 (batch_size, context_length)。
                                           y 是目标张量，形状为 (batch_size, context_length)。
    """
    
    #随机选择一个起点
    ix=torch.randint(0,len(dataset)-context_length,(batch_size,))
    
    #提取输入序列和对应目标序列
    x=torch.from_numpy(np.stack([dataset[i:i+context_length]for i in ix]))
    y=torch.from_numpy(np.stack([dataset[i+1:i+context_length] for i in ix]))
    
    #将张量移动到指定设备上
    
    x=x.to(device)
    y=y.to(device)
    
    return x,y