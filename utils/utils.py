import torch
import torch.optim as optim
import torch.nn as nn
import itertools
import numpy as np
import os
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import logging
from pytz import timezone
from datetime import datetime
import timm
import sys
import torchvision.transforms as T
import gc

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def imshow_np(img, filename):
    height, width, depth = img.shape
    if depth == 1:
        img = img[:, :, 0]
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("./save_image_oulu/dynamic/" + filename + ".png",
                bbox_inches='tight', pad_inches=0)
    plt.close()

def get_inf_iterator(data_loader):
    # """Inf data iterator."""
    while True:
        for images,  live_spoof_labels, material_label, live_map, spoof_map in data_loader:
            yield (images, live_spoof_labels, material_label, live_map, spoof_map)


def get_data_loader(data_path="", data_path2="live", amap_path="", batch_size=5, shuffle=True, drop_last=True):
    # data path
    data = None
    live_spoof_label = None
    live_related = None
    spoof_related = None
    material_label = None

    if data_path2 == "live":
        data = np.load(data_path)
        material_label = np.ones(len(data), dtype=np.int64)

        live_spoof_label = np.ones(len(data), dtype=np.int64)
        live_related = np.load(amap_path)
        spoof_related = np.zeros((len(data), 14, 14, 1), dtype=np.float32)
    else:
        print_data = np.load(data_path)
        print_data = print_data[int(len(print_data)*0.6):]
        replay_data = np.load(data_path2)
        replay_data = replay_data[int(len(replay_data)*0.6):]
        data = np.concatenate((print_data, replay_data), axis=0)

        print_lab = np.zeros(len(print_data), dtype=np.int64)
        replay_lab = np.ones(len(replay_data), dtype=np.int64) * 2
        material_label = np.concatenate((print_lab, replay_lab), axis=0)
        del print_lab, replay_lab, print_data, replay_data

        live_spoof_label = np.zeros(len(data), dtype=np.int64)
        live_related = np.zeros((len(data), 14, 14, 1), dtype=np.float32)
        spoof_related = np.load(amap_path)

    # dataset
    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data, (0, 3, 1, 2))),
                                              torch.tensor(live_spoof_label),
                                              torch.tensor(material_label),
                                              torch.tensor(np.transpose(live_related, (0, 3, 1, 2))),
                                              torch.tensor(np.transpose(spoof_related, (0, 3, 1, 2))))
    # free memory
    del data
    # del live_related
    # del spoof_related
    gc.collect()
    # dataloader
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader

def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def logger(results_filename):
    logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
    file_handler = logging.FileHandler(filename='/home/huiyu8794/TIP/logger/'+ results_filename +'_test.log')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    date = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)