from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import argparse

from transform import *
import utils

class CompositionTrainDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_list, self.label_list = utils.get_files(opt.train_root)
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
        img_path = self.img_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        transform = self.transformer_A.get_transform()
        img_t = transform(image=img)['image']
        # label read
        label = self.label_list[index]
        # img: 1 * 256 * 256; 
        return img_t, label

class CompositionValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.val_img_list, self.label_list = utils.get_files(opt.val_root)
        if opt.norm_mean and opt.norm_std:
            self.total_mean = opt.norm_mean
            self.total_std = opt.norm_std
        else:
            self.train_img_list, _ = utils.get_files(opt.train_root)
            self.total_mean, self.total_std = utils.cal_norm_paras(self.train_img_list)
            opt.norm_mean = self.total_mean
            opt.norm_std = self.total_std
        self.transformer_A = CompositionTestTransform_A(self.opt, self.total_mean, self.total_std)

    def __len__(self):
        return len(self.val_img_list)

    def __getitem__(self, index):
        # image read
        img_path = self.val_img_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        transform = self.transformer_A.get_transform()
        img_t = transform(image=img)['image']
        # label read
        label = self.label_list[index]
        # img: 1 * 256 * 256; 
        return img_t, label

class CompositionTestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.test_img_list, _ = utils.get_files(opt.test_root, False)
        if opt.norm_mean and opt.norm_std:
            self.total_mean = opt.norm_mean
            self.total_std = opt.norm_std
        else:
            self.train_img_list, _ = utils.get_files(opt.train_root, False)
            self.total_mean, self.total_std = utils.cal_norm_paras(self.train_img_list)
            opt.norm_mean = self.total_mean
            opt.norm_std = self.total_std
        self.transformer_A = CompositionTestTransform_A(self.opt, self.total_mean, self.total_std)

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, index):
        # image read
        img_path = self.test_img_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        transform = self.transformer_A.get_transform()
        img_t = transform(image=img)['image']
        # img: 1 * 256 * 256; 
        return img_t


if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--train_root', type = str, default = "./data/train", help = 'the base training folder')
    parser.add_argument('--val_root', type = str, default = "./data/val", help = 'the base validating folder')
    parser.add_argument('--test_root', type = str, default = "./data/val", help = 'the base testing folder')
    parser.add_argument('--comp_types_num', type = int, default = 9, help = 'composition types num')
    parser.add_argument('--norm_mean', type = float, default = 0.457, help = 'set None to recalculate')
    parser.add_argument('--norm_std', type = float, default = 0.210, help = 'set None to recalculate')
    parser.add_argument('--size_w', type = int, default = 256, help = 'size of image')
    parser.add_argument('--size_h', type = int, default = 256, help = 'size of image')
    opt = parser.parse_args()
    print(opt)
    
    train_dataset = CompositionTrainDataset(opt)
    val_dataset = CompositionValDataset(opt)
    test_dataset = CompositionTestDataset(opt)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for loader, loader_name in zip([train_loader, val_loader], ['Train', 'Validation']):
        for batch_idx, (data, label) in enumerate(loader):
            if batch_idx == 9:
                print(f"{loader_name} Batch Index: {batch_idx}")
                print(f"Label: {label}")

                data = (data * opt.norm_std) + opt.norm_mean
                data = data.squeeze(0).numpy()
                data = (data * 255).astype(np.uint8)
                
                cv2.imwrite(f"dev_samples/original_{loader_name.lower()}.png", data[0])

    for batch_idx, (data, _, _) in enumerate(test_loader):
        if batch_idx == 9:
            print(f"Test Batch Index: {batch_idx}")

            data = (data * opt.norm_std) + opt.norm_mean
            data = data.squeeze(0).numpy()
            data = (data * 255).astype(np.uint8)
            
            cv2.imwrite(f"dev_samples/original_test.png", data[0])