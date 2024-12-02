import torch.distributed as dist

def main():
    dist.init_process_group(
        backend="nccl", 
        init_method="tcp://127.0.0.1:12345", 
        world_size=2, 
        rank=int(os.environ["RANK"])
    )
    print(f"Rank {dist.get_rank()} initialized successfully.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
