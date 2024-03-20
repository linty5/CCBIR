from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os.path as osp
import random
import argparse

from transform import *
import utils

class RetrievalTrainDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_list = utils.get_files(opt.train_root)
        if opt.norm_mean and opt.norm_std:
            self.total_mean = opt.norm_mean
            self.total_std = opt.norm_std
        else:
            self.total_mean, self.total_std = utils.cal_norm_paras(self.img_list)
            opt.norm_mean = self.total_mean
            opt.norm_std = self.total_std
        self.transformer_A = CompositionTrainTransform_A(self.opt, self.total_mean, self.total_std)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # image read
        if index > len(self.img_list) - 2:
            print("Shot end")
            return -1, -1, -1
        img_origin_path = self.img_list[index]
        img_pos_path = self.img_list[index + 1]
        if img_pos_path.split('_')[-2] != img_origin_path.split('_')[-2] or img_pos_path.split('_')[-3] != img_origin_path.split('_')[-3]:
            print("Change shot")
            return -1, -1, -1
        img_neg_path = random.choice(self.img_list)
        # print(img_origin_path, img_pos_path, img_neg_path)
        while img_neg_path.split('_')[-2] == img_origin_path.split('_')[-2] and img_neg_path.split('_')[-3] == img_origin_path.split('_')[-3]:
            img_neg_path = random.choice(self.img_list)
            
        transform = self.transformer_A.get_transform()
        img_origin = cv2.imread(img_origin_path, cv2.IMREAD_GRAYSCALE)
        img_origin_t = transform(image=img_origin)['image']
        img_pos = cv2.imread(img_pos_path, cv2.IMREAD_GRAYSCALE)
        img_pos_t = transform(image=img_pos)['image']
        img_neg = cv2.imread(img_neg_path, cv2.IMREAD_GRAYSCALE)
        img_neg_t = transform(image=img_neg)['image']
        # print(img_origin_t, img_pos_t, img_neg_t)
        
        return img_origin_t, img_pos_t, img_neg_t

class RetrievalValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_list = utils.get_files(opt.val_root)
        if opt.norm_mean and opt.norm_std:
            self.total_mean = opt.norm_mean
            self.total_std = opt.norm_std
        else:
            self.train_img_list = utils.get_files(opt.train_root)
            self.total_mean, self.total_std = utils.cal_norm_paras(self.train_img_list)
            opt.norm_mean = self.total_mean
            opt.norm_std = self.total_std
        self.transformer_A = CompositionTrainTransform_A(self.opt, self.total_mean, self.total_std)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # image read
        if index > len(self.img_list) - 2:
            print("Shot end")
            return -1, -1, -1
        img_origin_path = self.img_list[index]
        img_pos_path = self.img_list[index + 1]
        if img_pos_path.split('_')[-2] != img_origin_path.split('_')[-2] or img_pos_path.split('_')[-3] != img_origin_path.split('_')[-3]:
            print("Change shot")
            return -1, -1, -1
        img_neg_path = random.choice(self.img_list)
        # print(img_origin_path, img_pos_path, img_neg_path)
        while img_neg_path.split('_')[-2] == img_origin_path.split('_')[-2] and img_neg_path.split('_')[-3] == img_origin_path.split('_')[-3]:
            img_neg_path = random.choice(self.img_list)
            
        transform = self.transformer_A.get_transform()
        img_origin = cv2.imread(img_origin_path, cv2.IMREAD_GRAYSCALE)
        img_origin_t = transform(image=img_origin)['image']
        img_pos = cv2.imread(img_pos_path, cv2.IMREAD_GRAYSCALE)
        img_pos_t = transform(image=img_pos)['image']
        img_neg = cv2.imread(img_neg_path, cv2.IMREAD_GRAYSCALE)
        img_neg_t = transform(image=img_neg)['image']
        # print(img_origin_t, img_pos_t, img_neg_t)
        
        return img_origin_t, img_pos_t, img_neg_t

class RetrievalTestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_list = utils.get_test_files(opt.test_root)
        if opt.norm_mean and opt.norm_std:
            self.total_mean = opt.norm_mean
            self.total_std = opt.norm_std
        else:
            self.train_img_list = utils.get_files(opt.train_root)
            self.total_mean, self.total_std = utils.cal_norm_paras(self.train_img_list)
            opt.norm_mean = self.total_mean
            opt.norm_std = self.total_std
        self.transformer_A = CompositionTrainTransform_A(self.opt, self.total_mean, self.total_std)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # image read
        # print(self.img_list)
        img_origin_path = self.img_list[index]
        img_origin_name = img_origin_path.split('/')[-1].split('.')[0]
        img_origin_prefix = img_origin_name.split('_')[0].split('\\')[-1] + '_' + img_origin_name.split('_')[1]
        img_origin_index = int(img_origin_name.split('_')[2])
        img_pos1_path = osp.join(osp.join(self.opt.test_root, 'imgs'), img_origin_prefix + '_' + str(img_origin_index - 1) + '.jpg')
        img_pos2_path = osp.join(osp.join(self.opt.test_root, 'imgs'), img_origin_prefix + '_' + str(img_origin_index + 1) + '.jpg')
        img_neg1_path = random.choice(self.img_list)
        # print(img_origin_path, img_pos_path, img_neg_path)
        while img_neg1_path.split('_')[-2] == img_origin_path.split('_')[-2] and img_neg1_path.split('_')[-3] == img_origin_path.split('_')[-3]:
            img_neg1_path = random.choice(self.img_list)
        img_neg2_path = random.choice(self.img_list)
        while (img_neg2_path.split('_')[-2] == img_origin_path.split('_')[-2] and img_neg2_path.split('_')[-3] == img_origin_path.split('_')[-3]) or \
        (img_neg2_path.split('_')[-2] == img_neg1_path.split('_')[-2] and img_neg2_path.split('_')[-3] == img_neg1_path.split('_')[-3]):
            img_neg2_path = random.choice(self.img_list)
            
        # print(img_origin_path, img_pos1_path, img_pos2_path, img_neg1_path, img_neg2_path)
            
        transform = self.transformer_A.get_transform()
        img_origin = cv2.imread(img_origin_path, cv2.IMREAD_GRAYSCALE)
        img_origin_t = transform(image=img_origin)['image']
        img_pos1 = cv2.imread(img_pos1_path, cv2.IMREAD_GRAYSCALE)
        img_pos1_t = transform(image=img_pos1)['image']
        img_pos2 = cv2.imread(img_pos2_path, cv2.IMREAD_GRAYSCALE)
        img_pos2_t = transform(image=img_pos2)['image']
        img_neg1 = cv2.imread(img_neg1_path, cv2.IMREAD_GRAYSCALE)
        img_neg1_t = transform(image=img_neg1)['image']
        img_neg2 = cv2.imread(img_neg2_path, cv2.IMREAD_GRAYSCALE)
        img_neg2_t = transform(image=img_neg2)['image']
        
        return img_origin_t, img_pos1_t, img_pos2_t, img_neg1_t, img_neg2_t

class RetrievalSingleTestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_list = utils.get_single_test_files(opt.test_root)
        self.total_mean = opt.norm_mean
        self.total_std = opt.norm_std
        self.transformer_A = CompositionTrainTransform_A(self.opt, self.total_mean, self.total_std)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # image read
        img_path = self.img_list[index]
        if '\\' in img_path:
            img_name = img_path.split('\\')[-1].split('.')[0]
        else:
            img_name = img_path.split('/')[-1].split('.')[0]
      
        transform = self.transformer_A.get_transform()
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_t = transform(image=img)['image']
        
        return img_t, img_name

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--train_root', type = str, default = "./data/train", help = 'the base training folder')
    parser.add_argument('--val_root', type = str, default = "./data/val", help = 'the base validating folder')
    parser.add_argument('--test_root', type = str, default = "./data/val", help = 'the base testing folder')
    parser.add_argument('--norm_mean', type = float, default = None, help = 'set None to recalculate')
    parser.add_argument('--norm_std', type = float, default = None, help = 'set None to recalculate')
    parser.add_argument('--size_w', type = int, default = 256, help = 'size of image')
    parser.add_argument('--size_h', type = int, default = 256, help = 'size of image')
    opt = parser.parse_args()
    print(opt)
    
    train_dataset = RetrievalTrainDataset(opt)
    val_dataset = RetrievalValDataset(opt)
    test_dataset = RetrievalTestDataset(opt)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for loader, loader_name in zip([train_loader, val_loader], ['Train', 'Validation']):
        for batch_idx, (img_origin, img_pos, img_neg) in enumerate(loader):
            # print(img_origin.numpy().shape)
            if img_origin.numpy().shape == 1:
                continue
            if batch_idx == 1:
                img_origin = (img_origin * opt.norm_std) + opt.norm_mean
                img_origin = img_origin.squeeze(0).numpy()
                img_origin = (img_origin * 255).astype(np.uint8)
                
                cv2.imwrite(f"dev_samples/{loader_name.lower()}_original.jpg", img_origin[0])
                
                img_pos = (img_pos * opt.norm_std) + opt.norm_mean
                img_pos = img_pos.squeeze(0).numpy()
                img_pos = (img_pos * 255).astype(np.uint8)
                
                cv2.imwrite(f"dev_samples/{loader_name.lower()}_positive.jpg", img_pos[0])
                
                img_neg = (img_neg * opt.norm_std) + opt.norm_mean
                img_neg = img_neg.squeeze(0).numpy()
                img_neg = (img_neg * 255).astype(np.uint8)
                
                cv2.imwrite(f"dev_samples/{loader_name.lower()}_negative.jpg", img_neg[0])
                
    for batch_idx, (img_origin_t, img_pos1_t, img_pos2_t, img_neg1_t, img_neg2_t) in enumerate(test_loader):
        if batch_idx == 9:
            break