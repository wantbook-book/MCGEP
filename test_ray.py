import ray

# 初始化Ray
ray.init()

# 定义一个远程函数
@ray.remote
def square(x):
    return x * x

# 测试Ray
if __name__ == "__main__":
    # 创建一组远程任务
    futures = [square.remote(i) for i in range(10)]

    # 收集任务结果
    results = ray.get(futures)

    print("平方计算结果: ", results)

    # 检查是否正确
    expected = [i * i for i in range(10)]
    if results == expected:
        print("Ray正常工作!")
    else:
        print("结果不一致，Ray可能有问题!")
    
    # 关闭Ray
    ray.shutdown()
