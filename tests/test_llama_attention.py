import torch
import os
import torch.distributed as dist

from vllm.model_executor.models.llama import LlamaAttention, LlamaConfig
from vllm.config import CacheConfig, QuantizationConfig

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29803'
dist.init_process_group(backend='xccl')

def test_llama_attention_forward_tp4(config):
    from vllm.model_executor.models import llama
    llama.get_tensor_model_parallel_world_size = dist.get_world_size()

    batch_size = 2
    seq_len = 5
    num_heads = 128
    num_kv_heads = 8
    hidden_size = 4096

    attn = LlamaAttention(
        config=config,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=10000,
        rope_scaling=None,
        max_position_embeddings=8192,
        quant_config=None,
        bias=False,
        bias_o_proj=False,
        cache_config=None,
        prefix="layers.0.attn",
    )

    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    output = attn(positions, hidden_states)

    # 检查形状
    assert output.shape == (batch_size, seq_len, hidden_size)
    # 检查数值类型
    assert output.dtype == hidden_states.dtype
    # 简单梯度检查
    output.sum().backward()
    for name, param in attn.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not get gradients"


if __name__ == '__main__':
    # 构造 LlamaConfig
    hidden_size = 4096
    num_heads = 128
    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        vocab_size=100,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.xpu.set_device(rank)
    test_llama_attention_forward_tp4(config)