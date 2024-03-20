import argparse
import tqdm
import os
import os.path as osp
import shutil
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import utils
from transform import CompositionTestTransform_A

def cal_cosine_similarity(A, B):
    similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity

def SingleRetriever(opt, net, com_net):
    net.eval()
    net = net.cuda()
    com_net.eval()
    com_net = com_net.cuda()
    
    transformer_A = CompositionTestTransform_A(opt, opt.norm_mean, opt.norm_std)
    transform = transformer_A.get_transform()
    target_img_list = os.listdir(opt.target_root)
    for target_img_name in target_img_list:
        target_img_path = osp.join(opt.target_root, target_img_name)
        target_img = np.expand_dims(cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE), axis=2)
        # print(target_img.shape)
        target_img_t = transform(image=target_img)['image'].unsqueeze(0)
        # print(target_img_t.shape)
        score_list = []
        results_dir = osp.join(opt.results_root, target_img_name)
        utils.check_path(results_dir)
        with torch.no_grad():
            test_dataset = dataset.RetrievalSingleTestDataset(opt)
            print('The overall number of testing images equals to %d' % len(test_dataset))
            test_loader = DataLoader(test_dataset, 
                                    batch_size = 1, 
                                    shuffle = False, 
                                    num_workers = 1)
            
            for _, (img, img_name) in enumerate(tqdm.tqdm(test_loader)):
                # print(img.shape)
                # Load and put to cuda
                img = img.cuda()                                           # out: [B, 1, 256, 256]
                target_img_t = target_img_t.cuda()
                # batch_length = img_origin.shape[0]
                
                _, kcm = com_net(img)
                _, kcm_target = com_net(target_img_t)
                # forward propagation
                out_img = net(img, kcm).cpu().numpy()[0]
                out_img_target = net(target_img_t, kcm_target).cpu().numpy()[0]
                
                # Compute losses
                score = cal_cosine_similarity(out_img, out_img_target)
                score_list.append([img_name, score])
        
        score_list.sort(key = lambda x: x[1], reverse = True)
        
        for i in range(10):
            img_name = score_list[i][0][0]
            origin_path = osp.join(opt.test_root, 'imgs', img_name + '.jpg')
            result_path = osp.join(results_dir, f"{i}_{img_name}.jpg")
            shutil.copy(origin_path, result_path)
        
        shutil.copy(target_img_path, osp.join(results_dir, 'target.jpg'))
            
        with open(osp.join(results_dir,  'score.txt'), 'w') as file:
            for item in score_list:
                # print(item)
                # print(osp.join(results_dir,  'score.txt'))
                file.write(str(item) + '\n')
    
    
if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--load_path', type = str, default = './models/model_name/cbirnet_best.pth', help = 'the load path for trained models')
    parser.add_argument('--load_com_path', type = str, default = './ccnet_model/ccnet_best.pth', help = 'the load path for trained models')
    parser.add_argument('--net_type', type = str, default = 'ImgRetriever', help = 'Used for composition classification')
    # Network parameters
    parser.add_argument('--loss_type', type = str, default = 'comb', help = 'type of loss function')
    parser.add_argument('--backbone_type', type = str, default = 'resnet50', help = 'choose the backbone')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--kcm_scale', type = int, default = 0.5, help = 'the scale of kcm')
    # Dataset parameters
    parser.add_argument('--test_root', type = str, default = "./data/selected", help = 'the base testing folder')
    parser.add_argument('--results_root', type = str, default = "./data/results_model_name", help = 'the base results folder')
    parser.add_argument('--target_root', type = str, default = "./data/target", help = 'the target image path')
    parser.add_argument('--comp_types_num', type = int, default = 9, help = 'composition types num')
    parser.add_argument('--norm_mean', type = float, default = 0.400, help = 'set None to recalculate')
    parser.add_argument('--norm_std', type = float, default = 0.147, help = 'set None to recalculate')
    parser.add_argument('--size_w', type = int, default = 256, help = 'size of image')
    parser.add_argument('--size_h', type = int, default = 256, help = 'size of image')
    opt = parser.parse_args()
    print(opt)
    
    net = utils.create_net(opt)
    net = net.cuda()
    com_net = utils.create_net(opt, is_com = True)
    com_net = com_net.cuda()
    SingleRetriever(opt, net, com_net)
