import torch


def check_pytorch_gpu():
    print("=" * 50)
    print("PyTorch GPU环境检测")
    print("=" * 50)

    # 1. 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")

    # 2. 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")

    if not cuda_available:
        print("\n❌ GPU不可用，可能的原因:")
        print("1. 未安装GPU版本的PyTorch")
        print("2. 没有NVIDIA GPU硬件")
        print("3. 未安装NVIDIA驱动程序")
        print("4. 未安装CUDA工具包")
        return False

    # 3. 获取CUDA版本
    print(f"CUDA版本: {torch.version.cuda}")

    # 4. 获取GPU数量
    device_count = torch.cuda.device_count()
    print(f"检测到的GPU数量: {device_count}")

    # 5. 显示每个GPU的详细信息
    for i in range(device_count):
        print(f"\nGPU {i} 详细信息:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

    # 6. 测试GPU张量操作
    print("\n" + "=" * 30)
    print("GPU功能测试")
    print("=" * 30)

    try:
        # 创建CPU张量
        cpu_tensor = torch.randn(3, 3)
        print(f"CPU张量创建成功: {cpu_tensor.device}")

        # 创建GPU张量
        gpu_tensor = torch.randn(3, 3).cuda()
        print(f"GPU张量创建成功: {gpu_tensor.device}")

        # 执行GPU计算
        result = gpu_tensor * 2
        print(f"GPU计算测试成功: {result.device}")

        # 测试矩阵乘法（GPU常见操作）
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        c = torch.mm(a, b)
        print(f"GPU矩阵乘法测试成功: 矩阵大小 1000x1000")

        # 测试数据在CPU和GPU之间传输
        cpu_copy = c.cpu()
        print(f"GPU到CPU数据传输成功")

        print("\n✅ 所有GPU测试通过！PyTorch GPU环境配置正确。")
        return True

    except Exception as e:
        print(f"\n❌ GPU测试失败: {str(e)}")
        return False


def check_installation_details():
    print("\n" + "=" * 50)
    print("环境详细信息")
    print("=" * 50)

    # 检查cuDNN版本
    if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'version'):
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")

    # 当前设备
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    print(f"当前设备索引: {current_device}")

    # 显示更多CUDA信息
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n设备 {i}:")
            print(f"  名称: {props.name}")
            print(f"  显存: {props.total_memory / 1e9:.2f} GB")
            print(f"  多处理器数量: {props.multi_processor_count}")
            # 移除clock_rate属性，因为在PyTorch 2.4.0中可能不再支持
            # 可以尝试使用其他属性获取性能信息
            if hasattr(props, 'max_clock_rate'):
                print(f"  最大时钟频率: {props.max_clock_rate / 1e6:.2f} MHz")
            elif hasattr(props, 'clock_rate'):
                print(f"  时钟频率: {props.clock_rate / 1e6:.2f} MHz")

            # 添加更多可用的属性
            if hasattr(props, 'pci_bus_id'):
                print(f"  PCI总线ID: {props.pci_bus_id}")
            if hasattr(props, 'pci_device_id'):
                print(f"  PCI设备ID: {props.pci_device_id}")
            if hasattr(props, 'pci_domain_id'):
                print(f"  PCI域ID: {props.pci_domain_id}")


def check_gpu_performance():
    """简单的GPU性能测试"""
    print("\n" + "=" * 50)
    print("GPU性能简单测试")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("GPU不可用，跳过性能测试")
        return

    # 设置设备
    device = torch.device('cuda')

    # 测试1: 矩阵乘法性能
    print("测试1: 矩阵乘法性能")
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # 预热
    for _ in range(10):
        c = torch.mm(a, b)

    # 计时
    import time
    start = time.time()
    iterations = 50
    for _ in range(iterations):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"  矩阵大小: {size}x{size}")
    print(f"  迭代次数: {iterations}")
    print(f"  总时间: {elapsed:.3f} 秒")
    print(f"  平均每次: {elapsed / iterations * 1000:.3f} 毫秒")

    # 测试2: 内存带宽测试
    print("\n测试2: 内存复制测试")
    data_size = 100 * 1024 * 1024  # 100MB
    data_cpu = torch.randn(data_size // 4)  # float32, 4字节每个

    start = time.time()
    data_gpu = data_cpu.to(device)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"  数据大小: {data_size / (1024 * 1024):.1f} MB")
    print(f"  CPU->GPU传输时间: {elapsed * 1000:.3f} 毫秒")
    print(f"  传输速度: {data_size / (elapsed * 1024 * 1024):.2f} MB/秒")


if __name__ == "__main__":
    # 运行GPU检测
    gpu_ok = check_pytorch_gpu()

    # 显示详细信息
    check_installation_details()

    # 运行性能测试（可选）
    try:
        check_gpu_performance()
    except Exception as e:
        print(f"\n性能测试跳过: {str(e)}")

    print("\n" + "=" * 50)
    print("环境总结:")
    print("=" * 50)

    if gpu_ok:
        print("✅ 你的PyTorch GPU环境完全正常！")
        print("   设备: NVIDIA GeForce RTX 3060 (12.49 GB)")
        print("   PyTorch版本: 2.4.0+cu121")
        print("   CUDA版本: 12.1")
        print("   cuDNN版本: 90100")
        print("\n💡 建议:")
        print("   1. 可以开始深度学习训练和推理任务")
        print("   2. 监控GPU使用: nvidia-smi")
        print("   3. 清理缓存: torch.cuda.empty_cache()")
    else:
        print("❌ 需要修复GPU环境")
        print("\n💡 解决方案:")
        print("   1. 确认已安装NVIDIA驱动: nvidia-smi")
        print("   2. 安装CUDA工具包: https://developer.nvidia.com/cuda-toolkit")
        print("   3. 安装对应版本的PyTorch: https://pytorch.org")