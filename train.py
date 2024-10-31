import torch
from models.distillation_net_learnable_low_level import *
from models. ViT_HV_DC_trip_atten import *
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
from loss.hard_triplet_loss import HardTripletLoss 
from loss.cosine_similarity import similar_cosine
import warnings
from utils.utils import *
warnings.filterwarnings("ignore")

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
Resize_224 = transforms.Resize((224, 224))

def produce_sample_level_amap(model=None, image_path="",image_path2="live", save_path="", target_category=None):
    image_data = None
    if image_path2=="live":
        image_data = np.load(image_path)
    else:
        print_data = np.load(image_path)
        replay_data = np.load(image_path2)
        image_data = np.concatenate((print_data, replay_data), axis=0) 
        del print_data, replay_data

    image_data = torch.tensor(np.transpose(image_data, (0, 3, 1, 2)))
    activation_map = np.zeros((len(image_data), 14, 14, 1), dtype=np.float32)
    trainset = torch.utils.data.TensorDataset(image_data)
    del image_data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)
    a_index = 0

    for i, data in enumerate(trainloader, 0):
        image = data[0]
        image = image.to(device_id)
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        grayscale_cam = cam(input_tensor=image, target_category=target_category)
        Resize = transforms.Resize([14, 14])
        grayscale_cam = Resize(torch.tensor(grayscale_cam))
        grayscale_cam = grayscale_cam.unsqueeze(3)

        for j in range(image.size(0)):
            activation_map[a_index] = grayscale_cam[j].cpu().detach().numpy() * 255
            a_index += 1

    np.save(save_path, activation_map)
    
    del activation_map
    gc.collect()

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
        replay_data = np.load(data_path2)
        data = np.concatenate((print_data, replay_data), axis=0)

        print_lab = np.zeros(len(print_data), dtype=np.int64)
        replay_lab = np.ones(len(replay_data), dtype=np.int64) * 2
        material_label = np.concatenate((print_lab, replay_lab), axis=0)

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
    import gc
    del data
    del live_related
    del spoof_related
    gc.collect()
    # dataloader
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader

def threshold_masking(catimg, live_amap, spoof_amap):
    half_size = int(catimg.shape[0] / 2)
    img_real, img_fake = catimg[:half_size], catimg[half_size:]
    live_la, spoof_la = live_amap[:half_size], live_amap[half_size:]
    live_sa, spoof_sa = spoof_amap[:half_size], spoof_amap[half_size:]
    mask_ratio = 0.6
    thres = 255 * 0.5
    img_real = Resize_224(img_real)
    img_fake = Resize_224(img_fake)

    random_mask = torch.rand(14, 14).to(device_id)
    random_mask[random_mask < mask_ratio] = 0
    random_mask[random_mask != 0] = 1

    ### for live image ###
    live_thres = live_la * random_mask.expand(live_la.shape[0],live_la.shape[1],-1,-1)
    live_thres[live_thres < thres] = 0
    live_thres[live_thres != 0] = 1
    live_la_mix = (live_la * (1-live_thres) + spoof_la * live_thres).to(device_id)
    live_sa_mix = (live_sa * (1-live_thres) + spoof_sa * live_thres).to(device_id)
    
    live_mask =  nn.functional.interpolate(live_thres, size=(224,224), mode="nearest")
    mix_live_img = (img_real * (1-live_mask) + img_fake * live_mask).to(device_id)

    ### for spoof image ###
    spoof_thres = spoof_sa * random_mask.expand(spoof_sa.shape[0],spoof_sa.shape[1],-1,-1)
    spoof_thres[spoof_thres < thres] = 0
    spoof_thres[spoof_thres != 0] = 1
    spoof_la_mix = (spoof_la * (1-spoof_thres) + live_la * spoof_thres).to(device_id)
    spoof_sa_mix = (spoof_sa * (1-spoof_thres) + live_sa * spoof_thres).to(device_id)

    spoof_mask =  nn.functional.interpolate(spoof_thres, size=(224,224), mode="nearest")
    mix_spoof_img = (img_fake * (1-spoof_mask) + img_real * spoof_mask).to(device_id)       

    catimg_mix = torch.cat([mix_live_img, mix_spoof_img], 0)
    live_amap_mix = torch.cat([live_la_mix, live_sa_mix], 0)
    spoof_amap_mix = torch.cat([spoof_la_mix, spoof_sa_mix], 0)

    return catimg_mix, live_amap_mix, spoof_amap_mix



device_id = "cuda:0" 
device_id1 = "cuda:1"
results_filename = 'HVDCformer_O' 
results_path = "/var/mplab_share_data/jxchong/TIP/" + results_filename

Amodel_path = "/var/mplab_share_data/jxchong/Amap_res18/RCM_O.tar"

os.system("rm -r " + results_path)

file_handler = logging.FileHandler(filename='/home/jxchong/TIP1/logger/'+results_filename+'_train.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

# replay casia Oulu MSU
dataset1 = "MSU"
dataset2 = "casia"
dataset3 = "replay"
logging.info(f"Train on {dataset1}, {dataset2}, {dataset3}")

live_path1 =        '/var/mplab_share_data/domain-generalization/' + dataset1 + '_images_live.npy'
live_path2 =        '/var/mplab_share_data/domain-generalization/' + dataset2 + '_images_live.npy'
live_path3 =        '/var/mplab_share_data/domain-generalization/' + dataset3 + '_images_live.npy'

print_path1 =       '/var/mplab_share_data/domain-generalization/' + dataset1 + '_print_images.npy'
print_path2 =       '/var/mplab_share_data/domain-generalization/' + dataset2 + '_print_images.npy'
print_path3 =       '/var/mplab_share_data/domain-generalization/' + dataset3 + '_print_images.npy'

replay_path1 =      '/var/mplab_share_data/domain-generalization/' + dataset1 + '_replay_images.npy'
replay_path2 =      '/var/mplab_share_data/domain-generalization/' + dataset2 + '_replay_images.npy'
replay_path3 =      '/var/mplab_share_data/domain-generalization/' + dataset3 + '_replay_images.npy'

live_amap_path1 =   '/var/mplab_share_data/jxchong/TIP/activation_map_results/activation_live_' + dataset1 + '.npy'
live_amap_path2 =   '/var/mplab_share_data/jxchong/TIP/activation_map_results/activation_live_' + dataset2 + '.npy'
live_amap_path3 =   '/var/mplab_share_data/jxchong/TIP/activation_map_results/activation_live_' + dataset3 + '.npy'

spoof_amap_path1 =  '/var/mplab_share_data/jxchong/TIP/activation_map_results/activation_spoof_' + dataset1 + '.npy'
spoof_amap_path2 =  '/var/mplab_share_data/jxchong/TIP/activation_map_results/activation_spoof_' + dataset2 + '.npy'
spoof_amap_path3 =  '/var/mplab_share_data/jxchong/TIP/activation_map_results/activation_spoof_' + dataset3 + '.npy'

Fas_Net = vit_base_patch16_224(pretrained=True).to(device_id)

AMapModel = ResNet_LS().to(device_id)
AMapModel.load_state_dict(torch.load(Amodel_path, map_location=device_id))
AMapModel.eval()
target_layers = [AMapModel.FeatExtor_LS.layer4[-1]]
cam = GradCAM(model=AMapModel, target_layers=target_layers, use_cuda=True)
     
produce_sample_level_amap(model=AMapModel, image_path=live_path1, save_path=live_amap_path1, target_category=1)
produce_sample_level_amap(model=AMapModel, image_path=live_path2, save_path=live_amap_path2, target_category=1)
produce_sample_level_amap(model=AMapModel, image_path=live_path3, save_path=live_amap_path3, target_category=1)
produce_sample_level_amap(model=AMapModel, image_path=print_path1, image_path2=replay_path1, save_path=spoof_amap_path1, target_category=0)
produce_sample_level_amap(model=AMapModel, image_path=print_path2, image_path2=replay_path2, save_path=spoof_amap_path2, target_category=0)
produce_sample_level_amap(model=AMapModel, image_path=print_path3, image_path2=replay_path3, save_path=spoof_amap_path3, target_category=0)

criterionCls = nn.CrossEntropyLoss().to(device_id)
criterionMSE = torch.nn.MSELoss().to(device_id1)
criterionCos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device_id)

lr = random.choice([1e-5])
optimizer_meta = optim.NAdam(Fas_Net.parameters(), lr=lr, betas=(0.8, 0.9))

batch_size = 20

logging.info(f"batch_size: {batch_size}")
logging.info(f"lr: {lr}")

criterion_trip_hard = HardTripletLoss(margin=0.1, hardest=True).to(device_id)
criterion_trip_soft = HardTripletLoss(margin=0.05, hardest=True).to(device_id)

Fas_Net.train()

model_save_step = 10
model_save_epoch = 1
save_index = 0

data1_real = get_data_loader(data_path=live_path1, data_path2="live", amap_path=live_amap_path1,
                             batch_size=batch_size, shuffle=True)
data2_real = get_data_loader(data_path=live_path2, data_path2="live", amap_path=live_amap_path2,
                             batch_size=batch_size, shuffle=True)
data3_real = get_data_loader(data_path=live_path3, data_path2="live", amap_path=live_amap_path3,
                             batch_size=batch_size, shuffle=True)
data1_fake = get_data_loader(data_path=print_path1, data_path2=replay_path1, amap_path=spoof_amap_path1,
                             batch_size=batch_size, shuffle=True)
data2_fake = get_data_loader(data_path=print_path2, data_path2=replay_path2, amap_path=spoof_amap_path2,
                             batch_size=batch_size, shuffle=True)
data3_fake = get_data_loader(data_path=print_path3, data_path2=replay_path3, amap_path=spoof_amap_path3,
                             batch_size=batch_size, shuffle=True)

iternum = max(len(data1_real), len(data2_real),
              len(data3_real), len(data1_fake),
              len(data2_fake), len(data3_fake))
log_step = 20
logging.info(f"iternum={iternum}")
data1_real = get_inf_iterator(data1_real)
data2_real = get_inf_iterator(data2_real)
data3_real = get_inf_iterator(data3_real)
data1_fake = get_inf_iterator(data1_fake)
data2_fake = get_inf_iterator(data2_fake)
data3_fake = get_inf_iterator(data3_fake)
 
T_transform = torch.nn.Sequential(
        T.Pad(40, padding_mode="symmetric"),
        T.RandomRotation(30), 
        T.RandomHorizontalFlip(p=0.5),
        T.CenterCrop(224),
)

for epoch in range(3):

    for step in range(iternum):
        # ============ one batch extraction ============#
        img1_real, ls_lab1_real, m_lab1_real, live_img1_liveMap, spoof_img1_liveMap = next(data1_real)
        img1_fake, ls_lab1_fake, m_lab1_fake, live_img1_fakeMap, spoof_img1_fakeMap = next(data1_fake)

        img2_real, ls_lab2_real, m_lab2_real, live_img2_liveMap, spoof_img2_liveMap = next(data2_real)
        img2_fake, ls_lab2_fake, m_lab2_fake, live_img2_fakeMap, spoof_img2_fakeMap = next(data2_fake)

        img3_real, ls_lab3_real, m_lab3_real, live_img3_liveMap, spoof_img3_liveMap = next(data3_real)
        img3_fake, ls_lab3_fake, m_lab3_fake, live_img3_fakeMap, spoof_img3_fakeMap = next(data3_fake)

        # ============ one batch collection ============#
        catimg = torch.cat([img1_real, img2_real, img3_real,
                            img1_fake, img2_fake, img3_fake], 0).to(device_id)
        ls_lab = torch.cat([ls_lab1_real, ls_lab2_real, ls_lab3_real,
                            ls_lab1_fake, ls_lab2_fake, ls_lab3_fake], 0).to(device_id)
        m_lab = torch.cat([m_lab1_real, m_lab2_real, m_lab3_real,
                           m_lab1_fake, m_lab2_fake, m_lab3_fake], 0).to(device_id)
        m_lab_mix = torch.cat([m_lab1_fake, m_lab2_fake, m_lab3_fake,
                               m_lab1_fake, m_lab2_fake, m_lab3_fake], 0).to(device_id)
        live_amap = torch.cat([live_img1_liveMap, live_img2_liveMap, live_img3_liveMap,
                               live_img1_fakeMap, live_img2_fakeMap, live_img3_fakeMap], 0).to(device_id)
        spoof_amap = torch.cat([spoof_img1_liveMap, spoof_img2_liveMap, spoof_img3_liveMap,
                                spoof_img1_fakeMap, spoof_img2_fakeMap, spoof_img3_fakeMap], 0).to(device_id)

        miximg, live_miximg, spoof_miximg = threshold_masking(catimg, live_amap, spoof_amap)

        # ============ one batch collection ============#

        batchidx = list(range(len(catimg)))
        random.shuffle(batchidx)

        img_rand = catimg[batchidx, :]
        ls_lab_rand = ls_lab[batchidx]
        m_lab_rand = m_lab[batchidx]
        liveGT_rand = live_amap[batchidx, :].to(device_id1)
        spoofGT_rand = spoof_amap[batchidx, :].to(device_id1)
		
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.random.manual_seed(seed)
        img_rand = T_transform(img_rand)
        
        pred, features, live_amap_pred, spoof_amap_pred = Fas_Net(NormalizeData_torch(img_rand)) 
        
        live_amap_pred = live_amap_pred.cuda(1)
        spoof_amap_pred = spoof_amap_pred.cuda(1)
        
        Loss_cls = criterionCls(pred.squeeze(), ls_lab_rand)

        random.seed(seed)
        torch.random.manual_seed(seed)
        Loss_dep = criterionMSE(live_amap_pred, liveGT_rand)
        
        random.seed(seed)
        torch.random.manual_seed(seed)
        Loss_dep += criterionMSE(spoof_amap_pred, spoofGT_rand)

        # ============ new added: masked dual attention supervision ============ #
        img_rand = miximg[batchidx, :]
        live_mix_rand = live_miximg[batchidx, :].to(device_id1)
        spoof_mix_rand = spoof_miximg[batchidx, :].to(device_id1)
        m_lab_mix_rand = m_lab_mix[batchidx]
        

        _, mix_feature, live_amap_pred, spoof_amap_pred = Fas_Net(NormalizeData_torch(img_rand))
        
        live_amap_pred = live_amap_pred.to(device_id1)
        spoof_amap_pred = spoof_amap_pred.to(device_id1)

        Loss_trip = criterion_trip_hard(features, m_lab_rand)
         
        # features = torch.cat([features, mix_feature], 1)
        # m_lab_rand = torch.cat([m_lab_rand, m_lab_mix_rand], 1)

        #################################################################################################################################################################
        live_features = torch.empty(1, 768).to(device_id)
        print_features = torch.empty(1, 768).to(device_id)
        replay_features = torch.empty(1, 768).to(device_id)

        for i,label in enumerate(m_lab_rand):
            if label == 1:
                live_features = torch.cat([live_features, features[i].unsqueeze(0)], 0)
            if label == 0:
                print_features = torch.cat([print_features, features[i].unsqueeze(0)], 0)
            if label == 2:
                replay_features = torch.cat([replay_features, features[i].unsqueeze(0)], 0)
        
        live_features = live_features[1:]
        print_features = print_features[1:]
        replay_features = replay_features[1:]
        
        del features,img_rand
        gc.collect()

        #################################################################################################################################################################

        print_mix_features = torch.empty(1, 768).to(device_id)
        replay_mix_features = torch.empty(1, 768).to(device_id)

        for i,label in enumerate(m_lab_mix_rand):
            if label == 0:
                print_mix_features = torch.cat([print_mix_features, mix_feature[i].unsqueeze(0)], 0)
            if label == 2:
                replay_mix_features = torch.cat([replay_mix_features, mix_feature[i].unsqueeze(0)], 0)
        
        print_mix_features = print_mix_features[1:]
        replay_mix_features = replay_mix_features[1:]

        live_mix_features = torch.cat([live_features, replay_mix_features, print_mix_features], 0)
        live_mix_labels = torch.cat([torch.zeros(live_features.shape[0]), torch.ones(replay_mix_features.shape[0]), torch.ones(print_mix_features.shape[0])*2], 0).to(device_id)

        Loss_trip_mix_print = criterion_trip_soft(live_mix_features, live_mix_labels)




        Loss_dep += criterionMSE(live_amap_pred, live_mix_rand) * 0.1
        Loss_dep += criterionMSE(spoof_amap_pred, spoof_mix_rand) * 0.1

        # print(print_mix_features[:len(print_features)].shape, live_features[:len(print_features)].shape, print_features.shape)
        Loss_similar = 1-torch.mean(criterionCos(replay_mix_features[:len(replay_features)]-live_features[:len(replay_features)], replay_features-live_features[:len(replay_features)]))
        Loss_similar+= 1-torch.mean(criterionCos(print_mix_features[:len(print_features)]-live_features[:len(print_features)], print_features-live_features[:len(print_features)]))

        # ============ new added: masked dual attention supervision ============ #

        Loss_all = Loss_cls + Loss_dep.to(device_id) * 0.004 + Loss_trip + Loss_trip_mix_print + Loss_similar * 0.1

        optimizer_meta.zero_grad()
        Loss_all.backward()
        optimizer_meta.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d]  Loss_cls %.8f Loss_trip %.8f Loss_dep %.8f Loss_trip_mix_print %.8f Loss_similar %.8f'
                  % (epoch, step, Loss_cls.item(), Loss_trip.item(), Loss_dep.item()*0.004, Loss_trip_mix_print.item(), Loss_similar.item()))
            
        if ((step + 1) % model_save_step == 0):
            mkdir(results_path)
            save_index += 1
            torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                          "FASNet-{}.tar".format(save_index)))

    if ((epoch + 1) % model_save_epoch == 0):
        mkdir(results_path)
        save_index += 1
        torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                      "FASNet-{}.tar".format(save_index)))
