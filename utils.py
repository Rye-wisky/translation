import torch
from torch import nn
import torch.nn.functional as F # <-- 导入 F
from typing import Dict, List, Tuple, Optional
from torch.optim.optimizer import Optimizer
from jaxtyping import Float, Int
from torch import Tensor
import os
import math # <-- 添加 math 导入 (为了 get_cosine_schedule_with_warmup)
from typing import BinaryIO, IO
from einops import rearrange,einsum

def softmax(x:Tensor,dim:int=-1)->Tensor:
    max_x=x.max(dim=dim,keepdim=True)[0] # <--- 在末尾添加 [0] 来获取 'values'
    x_stable=x-max_x
    exps=torch.exp(x_stable)
    sum_exp=exps.sum(dim=dim,keepdim=True)
    return exps/sum_exp

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
)->Float[Tensor, " ... queries d_v"]:
    #获取d_k
    d_k=Q.size(-1)
    
    #计算注意力分数
    score=einsum(Q,K,"... query d_k,... key d_k->... query key")
    scaled_score=score/math.sqrt(d_k)
    if mask is not None:
        scaled_score=scaled_score.masked_fill(mask==False,-torch.inf)
    attention_weights=softmax(scaled_score,dim=-1)

    # --- 添加这一行 ---
    # 将 attention_weights (float32) 转换回 V (float16) 的类型
    attention_weights = attention_weights.to(V.dtype)
    
    output=einsum(attention_weights,V,"... query key,... value d_k->... query d_k")
    return output

# ==============================================================================
# --- 修改后的 cross_entropy ---
# 我们使用 PyTorch 的官方实现，因为它支持 ignore_index (对于损失掩码至关重要)
# ==============================================================================
def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], 
    targets: Int[Tensor, " batch_size"],
    ignore_index: int = -100 # <-- 添加 ignore_index
) -> Float[Tensor, ""]:
    """
    计算交叉熵损失，支持 ignore_index。

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): 模型的原始 logits。
        targets (Int[Tensor, "batch_size"]): 目标 token ID。
        ignore_index (int, optional): 指定一个目标值，该值被忽略，不贡献给梯度。

    Returns:
        Float[Tensor, ""]: 平均交叉熵损失。
    """
    # 使用 PyTorch 的内置函数
    return F.cross_entropy(inputs, targets, ignore_index=ignore_index)
# ==============================================================================
# --- 原来的 cross_entropy 已被替换 ---
# ==============================================================================


class SGD(Optimizer):
    """
    实现了随机梯度下降（Stochastic Gradient Descent）优化器。
    该实现严格遵循 PyTorch Optimizer 的 API 规范。
    """
    def __init__(self, params, lr: float):
        """
        初始化 SGD 优化器。

        Args:
            params (iterable): 模型参数的可迭代对象，或定义了参数组的字典。
            lr (float): 学习率。
        """
        # 检查学习率是否有效
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")

        # 设置优化器的默认超参数
        defaults = dict(lr=lr)
        
        # 调用父类的构造函数
        # 这会处理参数组，并将 defaults 应用于每个组
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        执行单步优化。

        Args:
            closure (callable, optional): 一个可以重新评估模型并返回损失的闭包。
                                         对于大多数用例来说是可选的。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有参数组
        for group in self.param_groups:
            # 获取当前组的学习率
            lr = group['lr']
            
            # 遍历该组中的每个参数
            for p in group['params']:
                # 如果参数没有梯度，则跳过
                if p.grad is None:
                    continue
                
                # 获取梯度数据
                grad = p.grad
                
                # 应用 SGD 更新规则: p = p - lr * grad
                # p.data.add_() 是一个原地操作 (in-place)，效率更高
                # alpha=-lr 表示将 grad 乘以 -lr 再加到 p.data 上
                p.data.add_(grad, alpha=-lr)
        
        return loss


class AdamW(Optimizer):
    """
    实现了 AdamW 优化器。
    
    该算法在论文《Decoupled Weight Decay Regularization》中被提出。
    https://arxiv.org/abs/1711.05101
    """
    def __init__(self, 
                 params, 
                 lr: float = 1e-3, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2):
        """
        初始化 AdamW 优化器。

        Args:
            params (iterable): 模型参数的可迭代对象，或定义了参数组的字典。
            lr (float, optional): 学习率 (α)。默认为 1e-3。
            betas (Tuple[float, float], optional): 用于计算梯度及其平方的运行平均值的系数 (β₁, β₂)。
                                                 默认为 (0.9, 0.999)。
            eps (float, optional): 为增加数值稳定性而加到分母上的项 (ε)。默认为 1e-8。
            weight_decay (float, optional): 权重衰减系数 (λ)。默认为 1e-2。
        """
        # --- 输入验证 ---
        if not 0.0 <= lr:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"无效的 epsilon 值: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta 参数 beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta 参数 beta_2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"无效的权重衰减值: {weight_decay}")
            
        # 设置优化器的默认超参数
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # 调用父类的构造函数
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        执行单步优化。

        Args:
            closure (callable, optional): 一个可以重新评估模型并返回损失的闭包。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有参数组
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # 遍历该组中的每个参数
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW 不支持稀疏梯度，请考虑使用 SparseAdam。')

                # 获取该参数的状态
                state = self.state[p]

                # 延迟初始化状态 (Lazy state initialization)
                if len(state) == 0:
                    state['step'] = 0
                    # 一阶矩向量 (m)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 二阶矩向量 (v)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # 更新步数
                state['step'] += 1
                t = state['step']
                
                # --- AdamW 核心算法 ---
                
                # 1. 对参数应用权重衰减 (Decoupled Weight Decay)
                #    注意：这是 AdamW 与 Adam 的主要区别之一
                #    θ_t = θ_{t-1} - η * λ * θ_{t-1}
                p.mul_(1 - lr * weight_decay)

                # 2. 更新一阶矩估计 (m_t)
                #    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 3. 更新二阶矩估计 (v_t)
                #    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 4. 计算偏差修正项
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # 5. 计算修正后的学习率 α_t
                #    α_t = α * sqrt(1 - β₂^t) / (1 - β₁^t)
                step_size = lr / bias_correction1
                
                # 6. 计算分母
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                
                # 7. 更新参数
                #    θ_t = θ_t - α_t * (m_t / (sqrt(v_t) + ε))
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def get_cosine_schedule_with_warmup(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    根据 LLAMA 的训练方式，实现带热身的余弦学习率调度。

    该函数根据当前迭代步数 `it` 返回对应的学习率。

    Args:
        it (int): 当前的迭代步数 (t)。
        max_lr (float): 最大学习率 (α_max)。
        min_lr (float): 最小（最终）学习率 (α_min)。
        warmup_iters (int): 热身阶段的迭代次数 (T_w)。
        cosine_cycle_iters (int): 余弦退火周期的总迭代次数 (T_c)。
                                 注意：这应是训练的总步数。

    Returns:
        float: 当前迭代步数应使用的学习率。
    """
    # 1. 热身阶段 (Warm-up)
    # 在 T_w 步内，学习率从 0 线性增加到 max_lr
    if it < warmup_iters:
        return max_lr * it / warmup_iters

    # 2. 余弦退火阶段 (Cosine annealing)
    # 在 T_w 和 T_c 之间，学习率按余弦曲线从 max_lr 衰减到 min_lr
    if it < cosine_cycle_iters:
        # 计算在退火阶段的进度 (从 0 到 1)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # 计算余弦衰减因子 (从 1 到 0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 应用衰减
        return min_lr + (max_lr - min_lr) * cosine_decay

    # 3. 后退火阶段 (Post-annealing)
    # 在 T_c 之后，学习率保持为 min_lr
    return min_lr

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    保存模型、优化器和训练迭代的状态。

    参数:
        model (nn.Module): 需要保存状态的 PyTorch 模型。
        optimizer (optim.Optimizer): 需要保存状态的 PyTorch 优化器。
        iteration (int): 当前的训练迭代次数。
        out (Union[str, Path, BinaryIO]): 保存检查点文件的路径或文件对象。
    """
    #创建一个字典保存状态
    checkpoint={
        'iteration':iteration,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'model_args': model.model_args
    }   
    
    torch.save(checkpoint,out)
    print(f"检查点已保存至{out}(迭代次数：{iteration})")

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    从检查点文件加载模型和优化器的状态。

    参数:
        src (Union[str, Path, BinaryIO]): 检查点文件的路径或文件对象。
        model (nn.Module): 需要加载状态的 PyTorch 模型。
        optimizer (optim.Optimizer): 需要加载状态的 PyTorch 优化器。
        device (str): 要将张量加载到的设备 ('cpu', 'cuda', etc.)。

    返回:
        int: 从检查点中加载的训练迭代次数。
    """
    #创建一个字典读取状态
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    iteration=checkpoint['iteration']
    
    return iteration
