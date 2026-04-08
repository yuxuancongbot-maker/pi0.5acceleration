import torch
import time
import threading
import subprocess
import re

# 确认4090显卡是否被识别
assert torch.cuda.is_available(), "CUDA不可用，请检查显卡驱动和PyTorch安装"
device = torch.device("cuda:1")  # 4090通常是cuda:0（单卡场景）
print(f"使用显卡: {torch.cuda.get_device_name(device)}")

# 核心参数（针对4090调优，确保利用率稳定在50%左右）
BATCH_SIZE = 4096  # 单次计算的张量大小（4090显存足够，这个值适配50%利用率）
LOOP_INTERVAL = 0.0005  # 计算间隔（控制频率，避免利用率波动）
utilization_flag = True  # 控制计算线程的启停

def get_gpu_info(gpu_id=0):
    """
    兼容旧版PyTorch：通过nvidia-smi命令获取GPU利用率和显存信息
    :param gpu_id: GPU编号，默认0
    :return: (gpu_util: 利用率%, mem_used: 已用显存GB, mem_total: 总显存GB)
    """
    try:
        # 执行nvidia-smi命令获取GPU信息
        result = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        # 解析结果
        util, mem_used, mem_total = re.split(r",\s*", result.strip())
        return float(util), float(mem_used)/1024, float(mem_total)/1024  # 转GB
    except Exception as e:
        return 0.0, 0.0, 24.0  # 4090默认总显存24GB，异常时返回默认值

def gpu_compute_task():
    """GPU计算任务：循环执行矩阵运算，消耗GPU算力"""
    # 创建大张量并移到GPU（避免CPU-GPU数据传输干扰利用率）
    x = torch.randn(BATCH_SIZE, BATCH_SIZE, device=device)
    y = torch.randn(BATCH_SIZE, BATCH_SIZE, device=device)
    
    while utilization_flag:
        # 执行矩阵乘法（密集计算，消耗GPU算力）
        z = torch.matmul(x, y)
        # 同步计算（确保每次运算完成后再进行下一次，稳定利用率）
        torch.cuda.synchronize()
        # 短暂休眠，控制计算频率，避免利用率跑满
        time.sleep(LOOP_INTERVAL)

def monitor_gpu_utilization():
    """实时监控GPU利用率（兼容旧版PyTorch）"""
    while utilization_flag:
        # 通过nvidia-smi获取GPU信息（兼容所有版本）
        util, mem_used, mem_total = get_gpu_info(gpu_id=1)
        print(f"GPU利用率: {util:.1f}% | 显存使用: {mem_used:.2f}GB/{mem_total:.2f}GB", end="\r")
        time.sleep(0.5)  # 每0.5秒刷新一次

if __name__ == "__main__":
    try:
        # 启动GPU计算线程（核心：消耗50%算力）
        compute_thread = threading.Thread(target=gpu_compute_task)
        compute_thread.start()
        
        # 启动监控线程（查看利用率是否达标）
        monitor_thread = threading.Thread(target=monitor_gpu_utilization)
        monitor_thread.start()
        
        # 让程序持续运行，按Ctrl+C终止
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # 终止所有线程
        utilization_flag = False
        compute_thread.join()
        monitor_thread.join()
        torch.cuda.empty_cache()  # 释放GPU显存
        print("\n程序已终止，GPU资源已释放")