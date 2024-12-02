import os
import socket
import ray
import torch
from openrlhf.utils.distributed_util import init_process_group
from ray.util import placement_group
import vllm
from vllm import SamplingParams
from openrlhf.utils import DeepspeedStrategy
from abc import ABC
# @ray.remote(num_cpus=1, num_gpus=1)
class VllmGenerator(ABC):
    def __init__(self, strategy: DeepspeedStrategy, vllm_engines):
        super().__init__()
        self.strategy = strategy
        self.vllm_engines = vllm_engines

    
    def initialize_vllm_engines(self):
        """Ray Remote 函数，用于初始化 vLLM 引擎的分布式环境"""
        # breakpoint()
        address = ray._private.services.get_node_ip_address()
        os.environ['MASTER_ADDR'] = address.strip("[]")
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        strategy = self.strategy
        vllm_engines = self.vllm_engines   
        strategy.setup_distributed()
        if vllm_engines is not None and torch.distributed.get_rank() == 0:
            # 获取主节点 IP 地址
            master_address = ray._private.services.get_node_ip_address()
            # master_address = addr
            # master_port = port
            # 动态分配一个可用的端口
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            # 获取分布式策略的参数
            vllm_num_engines = strategy.args.vllm_num_engines
            vllm_tensor_parallel_size = strategy.args.vllm_tensor_parallel_size
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            # 设置后端类型
            backend = getattr(strategy.args, "vllm_sync_backend", "nccl")

            # 兼容 vLLM > 0.4.2 的情况，使用 gloo 后端
            if vllm.__version__ > "0.4.2" and os.getenv("NCCL_P2P_DISABLE", "0") == "0":
                backend = "gloo"
                print(
                    "Warning: using --vllm_sync_backend=gloo for vLLM version > 0.4.2 (or export NCCL_P2P_DISABLE=1)"
                )

            # 初始化每个 vLLM 引擎的进程组
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    "openrlhf",
                    backend=backend,
                )
                for i, engine in enumerate(vllm_engines)
            ]

            # 初始化主进程的进程组
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
            )

            # 等待所有引擎初始化完成
            ray.get(refs)

        # 分布式同步屏障
        torch.distributed.barrier()
    def generate_outputs(self, all_prompts: list[str], sampling_params):
        """
        Ray Remote 函数，用于在分布式环境中使用 vLLM 引擎生成响应。

        Args:
            vllm_engines (list): 所有可用的 vLLM 引擎。
            all_prompt_token_ids (list): 输入的 prompt token ID 列表。
            sampling_params (dict): 采样参数，用于控制生成。

        Returns:
            list: 所有生成的响应。
        """
        vllm_engines = self.vllm_engines
        # kwargs = {}
        # sampling_params = SamplingParams(
        #     temperature=kwargs.get("temperature", 1.0),
        #     top_p=kwargs.get("top_p", 1.0),
        #     top_k=kwargs.get("top_k", -1),
        #     max_tokens=kwargs.get("max_new_tokens", 1024),
        #     min_tokens=kwargs.get("min_new_tokens", 1),
        #     skip_special_tokens=kwargs.get("skip_special_tokens", False),
        # )

        # 获取当前进程的分布式 rank 和总进程数
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # 根据 rank 和 world_size 分配引擎
        if len(vllm_engines) <= world_size:
            llms = [vllm_engines[rank % len(vllm_engines)]]
        else:
            llms = vllm_engines[rank::world_size]

        # 根据引擎分配请求并生成响应
        all_output_refs = []
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompts = all_prompts[i * batch_size : (i + 1) * batch_size]
            if prompts:
                # 调用远程 vLLM 引擎生成响应
                all_output_refs.append(
                    # llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                    llm.generate.remote(prompts, sampling_params=sampling_params)
                )

        # 收集所有生成的响应
        all_outputs = sum(ray.get(all_output_refs), [])
        return all_outputs
