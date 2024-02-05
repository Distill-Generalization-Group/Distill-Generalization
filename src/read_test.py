import numpy as np
import torch

path="result/res_DC_CIFAR10_ConvNet_1ipc_1000.pt"
#load data from the path
data=torch.load(path)
syn_img=data['data'][0]
# syn_label=data['label']
print("data:", syn_img[0])
print("label:", syn_img[1])