from __future__ import print_function, division
import os
from einops import rearrange
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import utils
from pytorch_grad_cam import GradCAM
import torch.nn as nn
import torchvision.models as models


def get_frame(paths, transform_face):
    face_frames = []
    for path in paths:
        frame =  Image.open(path)
        face_frame = transform_face(frame)
        face_frames.append(face_frame)
    return face_frames


def getSubjects(configPath):
    
    f = open(configPath, "r")
    
    all_live, all_spoof = [], []
    while(True):
        line = f.readline()
        if not line:
            break
        line = line.strip()
        
        ls, subj = line.split(",")
        if(ls == "+1"):
            all_live.append(subj)
        else:
            all_spoof.append(subj)
    
    print(f"{configPath=}")
    print(f"{len(all_live)=}, {len(all_spoof)=}")
    
    return all_live, all_spoof


def produce_amap(model=None, image_data=None, ls="live"):
    target_category = None
    model.eval()
    target_layers = [model.FeatExtor_LS.layer4[-1]]
    activation_map = np.zeros((len(image_data), 14, 14, 1), dtype=np.float32)

    if ls=="live":
        target_category = 1
    else:
        target_category = 0
        
    # for i in range(len(image_data)):
    image = image_data.cuda()
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=image, target_category=target_category)
    Resize = transforms.Resize([14, 14])
    grayscale_cam = Resize(torch.tensor(grayscale_cam))
    grayscale_cam = grayscale_cam.unsqueeze(3)
    activation_map = grayscale_cam.cpu().detach().numpy() * 255
    activation_map = torch.Tensor(activation_map)
    activation_map = activation_map.permute(0, 3, 1, 2)
    return activation_map


class Oulu_Dataset(Dataset):
    def __init__(self, protocol=1, leave1out=None, seq_length=100, train_test_dev="Train", ls="live", amap_model=""):
        assert train_test_dev in ["Train", "Test", "Dev"]
        assert ls in ["live", "spoof"]
        assert protocol in [1, 2, 3, 4]

        self.root_dir = f"/var/mplab_share_data/OULU-NPU_all/" 
        if protocol == 1 or protocol == 2:
            assert leave1out is None
            self.config = os.path.join(self.root_dir, f"Protocols/Protocol_{protocol}/{train_test_dev}.txt")
            
        elif protocol == 3 or protocol == 4:
            assert leave1out in [1, 2, 3, 4, 5, 6]
            self.config = os.path.join(self.root_dir, f"Protocols/Protocol_{protocol}/{train_test_dev}_{leave1out}.txt")
        
        
        self.train_test_dev = train_test_dev
        self.seq_length = seq_length

        self.face_images = {}

        self.ls = ls
        opt_folder = "crop"
        
        all_live, all_spoof = getSubjects(self.config)
        if self.ls == "live":
            self.all_subjects = all_live
        else:
            self.all_subjects = all_spoof
        
        for subject in self.all_subjects:
            face_paths = sorted(glob.glob(os.path.join(self.root_dir, "all", subject, opt_folder, "*.jpg")))
            self.face_images[subject] = face_paths
        

        self.transform_face = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.amap_model = amap_model

    def __getitem__(self, idx):

        # get subject
        subject = self.all_subjects[idx]
        # get faces
        _face_paths = self.face_images[subject]
        
        # get faces
        # assume len(_face_paths) = 1000, then random select from [30, 1000]
        # say we selected #500 and seq_length = 100, then return 0500.jpg, 0501.jpg, ..., 0600.jpg
        if self.train_test_dev == "Train":
            start = random.randint(0, len(_face_paths) - self.seq_length)
            _face_paths = _face_paths[start : start+self.seq_length]
            
        _face_frame = torch.stack(get_frame(_face_paths, self.transform_face))

        # print(_face_frame.shape)

        if self.ls == "live":
            label = torch.ones(len(_face_frame), dtype=torch.int64)
            live_related = produce_amap(model=self.amap_model, image_data=_face_frame, ls='live')
            spoof_related = torch.zeros((len(_face_frame), 1, 14, 14), dtype=torch.float32)
            material_label= torch.zeros(len(_face_frame), dtype=torch.int64)
            
        else:
            label = torch.zeros(len(_face_frame), dtype=torch.int64)
            live_related = torch.zeros((len(_face_frame),1, 14, 14), dtype=torch.float32)
            spoof_related = produce_amap(model=self.amap_model, image_data=_face_frame, ls='spoof')
            attack_type_num = int(_face_paths[0][-15])
            assert attack_type_num in [2, 3, 4, 5]
            if (attack_type_num-1) % 5 == 2 or (attack_type_num-1) % 5 == 3:
                material_label= torch.ones(len(_face_frame), dtype=torch.int64)
            else:
                material_label= torch.ones(len(_face_frame), dtype=torch.int64)*2
            
        trainset = (_face_frame.clone().detach(),
                    label.clone().detach(),
                    material_label.clone().detach(),
                    live_related.clone().detach(),
                    spoof_related.clone().detach())

        return trainset


    def __len__(self):
        return len(self.all_subjects)



class SiW_Dataset(Dataset):
    def __init__(self, protocol=1, leave1out=None, seq_length=100, train_test_dev="Train", ls="live", amap_model=""):
        assert train_test_dev in ["Train", "Test"]
        assert ls in ["live", "spoof"]
        assert protocol in [1, 2, 3]

        self.root_dir = f"/var/mplab_share_data/SiW_all"

        if protocol == 1:
            assert leave1out is None
            self.config = os.path.join(self.root_dir, f"Protocols/Protocol_{protocol}/{train_test_dev}.txt")
            
        elif protocol == 2:
            assert leave1out in [1, 2, 3, 4]
            self.config = os.path.join(self.root_dir, f"Protocols/Protocol_{protocol}/{train_test_dev}_{leave1out}.txt")

        elif protocol == 3:
            assert leave1out in [1, 2]
            self.config = os.path.join(self.root_dir, f"Protocols/Protocol_{protocol}/{train_test_dev}_{leave1out}.txt")
        
        
        self.train_test_dev = train_test_dev
        self.seq_length = seq_length

        self.face_images = {}

        self.ls = ls

        all_live, all_spoof = getSubjects(self.config)
        if self.ls == "live":
            self.all_subjects = all_live
        else:
            self.all_subjects = all_spoof
        
        for subject in self.all_subjects:
            face_paths = sorted(glob.glob(os.path.join(self.root_dir, "all", train_test_dev, subject, "*.jpg")))
            self.face_images[subject] = face_paths
        

        self.transform_face = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        if train_test_dev == "Test":
            self.amap_model = None
        else:
            self.amap_model = amap_model

    def __getitem__(self, idx):

        # get subject
        subject = self.all_subjects[idx]
        # get faces
        _face_paths = self.face_images[subject]
        
        # get faces
        # assume len(_face_paths) = 1000, then random select from [30, 1000]
        # say we selected #500 and seq_length = 100, then return 0500.jpg, 0501.jpg, ..., 0600.jpg
        # if self.train_test_dev == "Train":
        start = random.randint(0, len(_face_paths) - self.seq_length)
        _face_paths = _face_paths[start : start+self.seq_length]
            
        _face_frame = torch.stack(get_frame(_face_paths, self.transform_face))
        
        if self.train_test_dev == "Train":
            if self.ls == "live":
                label = torch.ones(len(_face_frame), dtype=torch.int64)
                live_related = produce_amap(model=self.amap_model, image_data=_face_frame, ls='live')
                spoof_related = torch.zeros((len(_face_frame), 1, 14, 14), dtype=torch.float32)
                material_label= torch.zeros(len(_face_frame), dtype=torch.int64)
                
            else:
                label = torch.zeros(len(_face_frame), dtype=torch.int64)
                live_related = torch.zeros((len(_face_frame),1, 14, 14), dtype=torch.float32)
                spoof_related = produce_amap(model=self.amap_model, image_data=_face_frame, ls='spoof')
                segments = _face_paths[0].split('/')
                attack_type_num = int(segments[6][6])
                assert attack_type_num in [2, 3]
                if attack_type_num == 2:
                    material_label= torch.ones(len(_face_frame), dtype=torch.int64)
                else:
                    material_label= torch.ones(len(_face_frame), dtype=torch.int64)*2

            trainset = (_face_frame.clone().detach(),
                        label.clone().detach(),
                        material_label.clone().detach(),
                        live_related.clone().detach(),
                        spoof_related.clone().detach())
            
        else:
            if self.ls == "live":
                label = torch.ones(len(_face_frame), dtype=torch.int64)
                
            else:
                label = torch.zeros(len(_face_frame), dtype=torch.int64)

            trainset = (_face_frame.clone().detach(),
                        label.clone().detach(),
                        torch.zeros(len(_face_frame), dtype=torch.int64),
                        torch.zeros((len(_face_frame),1, 14, 14), dtype=torch.float32),
                        torch.zeros((len(_face_frame),1, 14, 14), dtype=torch.float32))

        return trainset


    def __len__(self):
        return len(self.all_subjects)


def get_loader(dataname='SiW', protocol=1, leave1out=None, seq_length=30, batch_size=1, shuffle=True, train_test_dev="Train", ls="live", amap_model=None):
    
    if dataname == 'SiW':
        _dataset = SiW_Dataset(protocol=protocol,
                            leave1out=leave1out,
                            seq_length=seq_length, 
                            train_test_dev=train_test_dev,
                            ls=ls,
                            amap_model=amap_model)
    
    elif dataname == 'Oulu':
        _dataset = Oulu_Dataset(protocol=protocol,
                                leave1out=leave1out,
                                seq_length=seq_length,
                                train_test_dev=train_test_dev,
                                ls=ls,
                                amap_model=amap_model)

    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True) 


def collate_batch(batch):
       
    img_list, ls_lab_real_list, m_lab_real_list, live_img_liveMap_list, spoof_img_liveMap_list = [], [], [], []
    
    for (img, ls_lab_real, m_lab_real, live_img_liveMap, spoof_img_liveMap) in batch:
        
        img_list.append(img)
        ls_lab_real_list.append(ls_lab_real)
        m_lab_real_list.append(m_lab_real)
        live_img_liveMap_list.append(live_img_liveMap)
        spoof_img_liveMap_list.append(spoof_img_liveMap)

    return img_list, ls_lab_real_list, m_lab_real_list, live_img_liveMap_list, spoof_img_liveMap_list


def imshow_np(img, filename):
    height, width, depth = img.shape
    if depth == 1:
        img = img[:, :, 0]
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("/home/huiyu8794/TIP/loader_test/" + filename + ".png",
                bbox_inches='tight', pad_inches=0)
    plt.close()


class ResNet_LS(nn.Module):
    def __init__(self, ):
        super(ResNet_LS, self).__init__()
        # #distangle_init_LS_with_ID
        self.FeatExtor_LS = models.resnet34(pretrained=True)
        self.FC_LS = torch.nn.Linear(1000, 2)

    def forward(self, x):
        if self.training:
            lsf = self.FeatExtor_LS(x)
            x_ls = self.FC_LS(lsf)
            return x_ls, lsf.detach()
        else:
            lsf = self.FeatExtor_LS(x)
            x_ls = self.FC_LS(lsf)
            return x_ls
        

def squeeze_all(ls_attribute):
    img_real, ls_lab, m_lab, img_liveMap, img_fakeMap = ls_attribute
    img_real = rearrange(img_real, 'd0 d1 d2 d3 d4 -> (d0 d1) d2 d3 d4')
    img_liveMap = rearrange(img_liveMap, 'd0 d1 d2 d3 d4 -> (d0 d1) d2 d3 d4')
    img_fakeMap = rearrange(img_fakeMap, 'd0 d1 d2 d3 d4 -> (d0 d1) d2 d3 d4')
    ls_lab = rearrange(ls_lab, 'd0 d1 -> (d0 d1)')
    m_lab = rearrange(m_lab, 'd0 d1 -> (d0 d1)')

    return img_real, ls_lab, m_lab, img_liveMap, img_fakeMap


def get_inf_iterator(data_loader):
    # """Inf data iterator."""
    while True:
        for images,  live_spoof_labels, material_label, live_map, spoof_map in data_loader:
            yield (images, live_spoof_labels, material_label, live_map, spoof_map)

if __name__ == "__main__":

    Amodel_path = '/shared/huiyu8794/Amap_res18/oulu_p1.tar'
    AMapModel = ResNet_LS().cuda()
    AMapModel.load_state_dict(torch.load(Amodel_path, map_location='cuda:0'))
        
    data_real = get_loader(protocol=1, leave1out=None, seq_length=5, batch_size=1, 
                           shuffle=True, train_test_dev="Train", ls="live", amap_model=AMapModel)
    
    data_fake = get_loader(protocol=1, leave1out=None, seq_length=5, batch_size=1, 
                           shuffle=True, train_test_dev="Train", ls="spoof", amap_model=AMapModel)
    
    iternum = max(len(data_real), len(data_fake))
    data_real = get_inf_iterator(data_real)
    data_fake = get_inf_iterator(data_fake)
   
    for i in range(10):
        real_attribute = next(data_real)
        fake_attribute = next(data_fake)

        img_real, ls_lab_real, m_lab_real, live_img_liveMap, spoof_img_liveMap = squeeze_all(real_attribute)
        img_fake, ls_lab_fake, m_lab_fake, live_img_fakeMap, spoof_img_fakeMap = squeeze_all(fake_attribute)

        # expect: batch x channel x height x width [a, 3, 256, 256]
        img_real = img_real[0].squeeze(0)
        live_img_liveMap = live_img_liveMap[0].squeeze(0)
        spoof_img_liveMap = spoof_img_liveMap[0].squeeze(0)

        img_fake = img_fake[0].squeeze(0)
        live_img_fakeMap = live_img_fakeMap[0].squeeze(0)
        spoof_img_fakeMap = spoof_img_fakeMap[0].squeeze(0)

        imshow_np(np.transpose(img_real.cpu().detach().numpy(), (1,2,0)) ,str(i)+"_real")
        imshow_np((live_img_liveMap.cpu().detach().numpy()) ,str(i)+"_live_real")
        imshow_np((spoof_img_liveMap.cpu().detach().numpy()) ,str(i)+"_spoof_real")

        imshow_np(np.transpose(img_fake.cpu().detach().numpy(), (1,2,0)) ,str(i)+"_fake")
        imshow_np((live_img_fakeMap.cpu().detach().numpy()) ,str(i)+"_live_fake")
        imshow_np((spoof_img_fakeMap.cpu().detach().numpy()) ,str(i)+"_spoof_fake")
        break
