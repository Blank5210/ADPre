import torch

print(torch.cuda.is_available())  # True 表示可以使用 GPU
print(torch.version.cuda)         # PyTorch 编译时使用的 CUDA 版本
print(torch.cuda.current_device())# 当前 GPU 设备索引
print(torch.cuda.get_device_name(0))  # 当前 GPU 名称
print(torch.__version__)