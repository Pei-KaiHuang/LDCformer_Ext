import torch
from models. ViT_HV_DC_trip_atten import *
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import logging
from pytz import timezone
from datetime import datetime
import sys
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
from utils.utils import *


dataset = 'Oulu' 
model_save_name = 'HVDCformer_O' 
results_path = "/var/mplab_share_data/jxchong/TIP/" + model_save_name + "/"
batch_size = 10


live_path =     '/var/mplab_share_data/domain-generalization/' + dataset + '_images_live.npy'
spoof_path =    '/var/mplab_share_data/domain-generalization/' + dataset + '_images_spoof.npy'


live_data = np.load(live_path)
spoof_data = np.load(spoof_path)
live_label = np.ones(len(live_data), dtype=np.int64)
spoof_label = np.zeros(len(spoof_data), dtype=np.int64)

total_data = np.concatenate((live_data, spoof_data), axis=0)
del live_data, spoof_data
total_label = np.concatenate((live_label, spoof_label), axis=0)
del live_label, spoof_label

trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                          torch.tensor(total_label))
# dataloader
data_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=False)


def logger(root_dir, results_filename, train=True):
    logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
    mkdir(root_dir)
    if train:
        file_handler = logging.FileHandler(filename= root_dir + results_filename +'_train.log')
    else:
        file_handler = logging.FileHandler(filename= root_dir + results_filename +'_test.log')

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    date = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
    
device_id = 'cuda:0'
logger('/home/jxchong/TIP1/logger/', model_save_name, train=False)
FASNet = vit_base_patch16_224(pretrained=True).to(device_id)

import glob
length = len(glob.glob(results_path + '*.tar'))
record = [1,100,100,100,100]
for epoch in reversed(range(1, length+1)):
    MataNet_path =  results_path + 'FASNet-' + str(epoch) + '.tar'
    FASNet.load_state_dict(torch.load(MataNet_path, map_location=device_id))
    FASNet.eval()
    score_list_ori = []
    Total_score_list_cs = []
    label_list = []
    TP = 0.0000001
    TN = 0.0000001
    FP = 0.0000001
    FN = 0.0000001

    TP_all = 0.0000001
    TN_all = 0.0000001
    FP_all = 0.0000001
    FN_all = 0.0000001
    for i, data in enumerate(data_loader, 0):
        # print(i)
        images, labels = data
        images = images.float().to(device_id)
        Resize = transforms.Resize([224, 224])
        label_pred,_,_,_ = FASNet(Resize(NormalizeData_torch(images))) 
        score = F.softmax(label_pred, dim=1).cpu().data.numpy()[:, 1]
        
        for j in range(images.size(0)): 
            score_list_ori.append(score[j])
            label_list.append(labels[j]) 
         
    score_list_ori = NormalizeData(score_list_ori)

    fpr, tpr, thresholds_cs = metrics.roc_curve(label_list, score_list_ori)
    threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)

    for i in range(len(score_list_ori)):
        score = score_list_ori[i]
        if (score >= threshold_cs and label_list[i] == 1):
            TP += 1
        elif (score < threshold_cs and label_list[i] == 0):
            TN += 1
        elif (score >= threshold_cs and label_list[i] == 0):
            FP += 1
        elif (score < threshold_cs and label_list[i] == 1):
            FN += 1

    APCER = FP / (TN + FP)
    NPCER = FN / (FN + TP)
    
    if record[1]>((APCER + NPCER) / 2):
            record[0]=epoch
            record[1]=((APCER + NPCER) / 2)
            record[2]=roc_auc_score(label_list, score_list_ori)
            record[3]=APCER
            record[4]=NPCER

    logging.info('[epoch %d]  APCER %.4f  NPCER %.4f  ACER %.4f  AUC %.4f'
            % (epoch, APCER, NPCER, np.round((APCER + NPCER) / 2, 4), np.round(roc_auc_score(label_list, score_list_ori), 4)))
            
            
logging.info(f"BEST Epoch {str(record[0])} ACER {str(record[1])} AUC {str(record[2])} APCER {str(record[3])} NPCER {str(record[4])} ")
print("Epoch_"   + str(record[0])  +"_CL_",
      "ACER_"    + str(record[1]) +
      "_AUC_"     + str(record[2]) +
      "_APCER_"   + str(record[3]) +
      "_NPCER_"    + str(record[4]))

