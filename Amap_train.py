import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from models.distillation_net_learnable_low_level import ResNet_LS
import argparse
import warnings
import gc
from utils.utils import *
warnings.filterwarnings("ignore")

device_id = 'cuda:0'

model_save_name = 'RCM_O'
model_save_path = "/shared/jxchong/Amap_res18/"

dataset1 = 'MSU'
dataset2 = 'replay'
dataset3 = 'casia'
# image shape: torch.Size([3, 256, 256])

live_data1 = np.load('/shared/domain-generalization/' + dataset1 + '_images_live.npy')
live_data2 = np.load('/shared/domain-generalization/' + dataset2 + '_images_live.npy')
live_data3 = np.load('/shared/domain-generalization/' + dataset3 + '_images_live.npy')

print_data1 = np.load('/shared/domain-generalization/' + dataset1 + '_print_images.npy')
print_data2 = np.load('/shared/domain-generalization/' + dataset2 + '_print_images.npy')
print_data3 = np.load('/shared/domain-generalization/' + dataset3 + '_print_images.npy')

replay_data1 = np.load('/shared/domain-generalization/' + dataset1 + '_replay_images.npy')
replay_data2 = np.load('/shared/domain-generalization/' + dataset2 + '_replay_images.npy')
replay_data3 = np.load('/shared/domain-generalization/' + dataset3 + '_replay_images.npy')

live_data = np.concatenate((live_data1, live_data2, live_data3), axis=0)
print_data = np.concatenate((print_data1, print_data2, print_data3), axis=0)
replay_data = np.concatenate((replay_data1, replay_data2, replay_data3), axis=0)


# image shape: torch.Size([3, 256, 256])

live_data = np.concatenate((live_data1, live_data2, live_data3), axis=0)
print_data = np.concatenate((print_data1, print_data2, print_data3), axis=0)
replay_data = np.concatenate((replay_data1, replay_data2, replay_data3), axis=0)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def imshow_np(img, filename):
    height, width, depth = img.shape
    if depth == 1:
        img = img[:, :, 0]
    plt.imshow(img)
    plt.savefig(filename)
    plt.close()

mkdir(model_save_path)
print(model_save_path)

spoof_data = np.concatenate((print_data, replay_data), axis=0)


gc.collect()
live_label = np.ones(len(live_data), dtype=np.int64)
spoof_label = np.zeros(len(spoof_data), dtype=np.int64)

total_data = np.concatenate((live_data, spoof_data), axis=0)
del live_data
del spoof_data
gc.collect()
total_label = np.concatenate((live_label, spoof_label), axis=0)
del live_label
del spoof_label
gc.collect()

trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                          torch.tensor(total_label))
del total_data, total_label
gc.collect()

# dataloader
trainloader_D = torch.utils.data.DataLoader(trainset,
                                            batch_size=60,
                                            shuffle=True)
del trainset
gc.collect()

Amodel = ResNet_LS().to(device_id)
criterionCls = nn.CrossEntropyLoss().to(device_id)
optimizer = torch.optim.Adam(Amodel.parameters(), lr=0.0001, eps=1e-08)
Amodel.train()
print("Start training")
for epoch in range(1, 100):
    count = 0
    total_loss = 0

    for i, data in enumerate(trainloader_D, 0):
        count += 1
        images, labels = data
        images = images.to(device_id)
        labels = labels.to(device_id)

        ls_pred,_ = Amodel(NormalizeData_torch(images))
        cls_loss = criterionCls(ls_pred, labels)
        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()
        total_loss += cls_loss.item()

    print('[epoch %d] Loss_cls %.5f' % (epoch, total_loss / count))
          
torch.save(Amodel.state_dict(), model_save_path + model_save_name + ".tar")
