"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm._C import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl
logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
LFCache = Tuple[torch.Tensor, torch.Tensor]

_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]

class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # Initialize the cache.
        self.kv_gpu_cache, self.lf_gpu_cache = self.allocate_gpu_cache()
        self.kv_cpu_cache, self.lf_cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:   #
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )
    
    def get_lf1_block_shape(self) -> Tuple[int, int, int]: #Can past_lf1 & past_lf2 set same block shape?
        return (
            self.num_heads * self.head_size,
            1, 1
            )

    def get_lf2_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads * self.head_size // 2,
            1, 1
            )


    def allocate_gpu_cache(self) -> Tuple[List[KVCache], List[LFCache]]:
        KV_gpu_cache: List[KVCache] = []
        LF_gpu_cache: List[LFCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        lf1_block_shape = self.get_lf1_block_shape()
        lf2_block_shape = self.get_lf2_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            lf1_blocks = torch.empty(
                size=(self.num_gpu_blocks, *lf1_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            lf2_blocks = torch.empty(
                size=(self.num_gpu_blocks, *lf2_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            KV_gpu_cache.append((key_blocks, value_blocks))
            LF_gpu_cache.append((lf1_blocks, lf2_blocks))
        return (KV_gpu_cache, LF_gpu_cache)

    def allocate_cpu_cache(self) -> Tuple[List[KVCache], List[LFCache]]:
        cpu_cache: Tuple[List[KVCache], List[LFCache]] = []
        KV_cpu_cache: List[KVCache] = []
        LF_cpu_cache: List[LFCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        lf1_block_shape = self.get_lf1_block_shape()
        lf2_block_shape = self.get_lf2_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            lf1_blocks = torch.empty(
                size=(self.num_gpu_blocks, *lf1_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            lf2_blocks = torch.empty(
                size=(self.num_gpu_blocks, *lf2_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            KV_cpu_cache.append((key_blocks, value_blocks))
            LF_cpu_cache.append((lf1_blocks, lf2_blocks))
        return (KV_cpu_cache, LF_cpu_cache)

    def _swap(
        self,
        src: Tuple[List[KVCache], List[LFCache]],
        dst: Tuple[List[KVCache], List[LFCache]],
        src_to_dst: Dict[int, int],  #Need modification?
    ) -> None:
        print('_swap', src_to_dst)
        exit()
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_value_cache = src[0][i]
                src_lf_cache = src[1][i]
                src_key_cache, src_value_cache = src_key_value_cache
                dst_key_value_cache = dst[0][i]
                dst_lf_cache = dst[1][i]
                dst_key_cache, dst_value_cache = dst_key_value_cache
                dst_lf1_cache, dst_lf2_cache = dst_lf_cache
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)
                cache_ops.swap_blocks(src_lf1_cache, dst_lf1_cache, src_to_dst)
                cache_ops.swap_blocks(src_lf2_cache, dst_lf2_cache, src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:  
        self._swap(self.kv_cpu_cache, self.kv_gpu_cache, src_to_dst)
        self._swap(self.lf_cpu_cache, self.lf_gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:  
        self._swap(self.kv_gpu_cache, self.kv_cpu_cache, src_to_dst)
        self._swap(self.lf_gpu_cache, self.lf_cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None: #Need modifcation?
        kv_caches = [kv_cache for kv_cache in self.kv_gpu_cache]
        lf_caches = [lf_cache for lf_cache in self.lf_gpu_cache]
        key_caches = [key_cache for key_cache, _ in kv_caches]
        value_caches = [value_cache for _, value_cache in kv_caches]

        lf1_caches = [lf1_cache for lf1_cache, _ in lf_caches]
        lf2_caches = [lf2_cache for _, lf2_cache in lf_caches]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)
        cache_ops.copy_blocks(lf1_caches, lf2_caches, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        lf1_cache_block = block_size * num_heads * head_size
        lf2_cache_block = block_size * num_heads * head_size // 2
        total = num_layers * (key_cache_block + value_cache_block + lf1_cache_block + lf2_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
