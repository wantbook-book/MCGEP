import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import deepspeed

def setup(rank, world_size):
    """使用 DeepSpeed 初始化分布式环境，并输出调试信息"""
    deepspeed.init_distributed(dist_backend="nccl")
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Distributed environment initialized.")
    print(f"[Rank {rank}] World size: {world_size}")
    print(f"[Rank {rank}] Backend: nccl")
    print(f"[Rank {rank}] Current device: {torch.cuda.current_device()}")

def train(rank, world_size):
    """分布式训练逻辑"""
    print(f"Rank {rank}/{world_size} starting training...")
    setup(rank, world_size)

    # 定义数据集和分布式采样器
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # 定义模型
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(rank)

    # 使用 DeepSpeed 优化器和引擎
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": 64,
            "gradient_accumulation_steps": 1,
            "fp16": {"enabled": False}
        }
    )

    # 训练循环
    for epoch in range(5):
        model_engine.train()
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(rank)
            target = target.to(rank)

            optimizer.zero_grad()
            output = model_engine(data)
            loss = nn.CrossEntropyLoss()(output, target)
            model_engine.backward(loss)
            model_engine.step()

            if rank == 0 and batch_idx % 10 == 0:  # 仅主进程打印日志
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")

def main():
    """主函数，使用 torch.multiprocessing 启动分布式训练"""
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    main()
