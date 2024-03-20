import argparse
import tqdm
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import utils

def cal_cosine_similarity(A, B):
    similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity

def Retriever(opt, net, com_net):
    net.eval()
    net = net.cuda()
    com_net.eval()
    com_net = com_net.cuda()
    
    count = 0
    sum = 0
    
    pos1_score_list = []
    pos2_score_list = []
    neg1_score_list = []
    neg2_score_list = []
    
    with torch.no_grad():
        test_dataset = dataset.RetrievalTestDataset(opt)
        print('The overall number of testing images equals to %d' % len(test_dataset))
        test_loader = DataLoader(test_dataset, 
                                batch_size = 1, 
                                shuffle = False, 
                                num_workers = 1)
        
        for _, (img_origin, img_pos1, img_pos2, img_neg1, img_neg2) in enumerate(tqdm.tqdm(test_loader)):
            # if img_origin.numpy().shape == (1,):
            #     print("Skipping batch due to empty batch")
            #     continue
            
            # Load and put to cuda
            img_origin = img_origin.cuda()                                           # out: [B, 1, 256, 256]
            img_pos1 = img_pos1.cuda()                                                # out: [B, 1, 256, 256]
            img_pos2 = img_pos2.cuda()                                                # out: [B, 1, 256, 256]
            img_neg1 = img_neg1.cuda()                                                # out: [B, 1, 256, 256]
            img_neg2 = img_neg2.cuda()                                                # out: [B, 1, 256, 256]
            
            # batch_length = img_origin.shape[0]
            
            _, kcm_origin = com_net(img_origin)
            _, kcm_pos1 = com_net(img_pos1)
            _, kcm_pos2 = com_net(img_pos2)
            _, kcm_neg1 = com_net(img_neg1)
            _, kcm_neg2 = com_net(img_neg2)

            # forward propagation
            out_img_origin = net(img_origin, kcm_origin).cpu().numpy()[0]
            out_img_pos1 = net(img_pos1, kcm_pos1).cpu().numpy()[0]
            out_img_pos2 = net(img_pos2, kcm_pos2).cpu().numpy()[0]
            out_img_neg1 = net(img_neg1, kcm_neg1).cpu().numpy()[0]
            out_img_neg2 = net(img_neg2, kcm_neg2).cpu().numpy()[0]
            
            # Compute losses
            score = 0
            pos1_score = cal_cosine_similarity(out_img_origin, out_img_pos1)
            pos2_score = cal_cosine_similarity(out_img_origin, out_img_pos2)
            neg1_score = cal_cosine_similarity(out_img_origin, out_img_neg1)
            neg2_score = cal_cosine_similarity(out_img_origin, out_img_neg2)
            print("pos1_score: ", pos1_score, " pos2_score: ", pos2_score, " neg1_score: ", neg1_score, " neg2_score: ", neg2_score)
            pos1_score_list.append(pos1_score)
            pos2_score_list.append(pos2_score)
            neg1_score_list.append(neg1_score)
            neg2_score_list.append(neg2_score)
            
            if pos1_score > neg1_score:
                score += 1
            if pos1_score > neg2_score:
                score += 1
            if pos2_score > neg1_score:
                score += 1
            if pos2_score > neg2_score:
                score += 1
            
            count += 1
            sum += score
    
    utils.check_path(opt.test_results_root)
    
    with open(osp.join(opt.test_results_root,  'pos1_score.txt'), 'w') as file:
        for item in pos1_score_list:
            file.write(str(item) + '\n')
    with open(osp.join(opt.test_results_root,  'pos2_score.txt'), 'w') as file:
        for item in pos2_score_list:
            file.write(str(item) + '\n')
    with open(osp.join(opt.test_results_root,  'neg1_score.txt'), 'w') as file:
        for item in neg1_score_list:
            file.write(str(item) + '\n')
    with open(osp.join(opt.test_results_root,  'neg2_score.txt'), 'w') as file:
        for item in neg2_score_list:
            file.write(str(item) + '\n')
    
    outputs = sum / count
    return outputs, count
    
if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--load_path', type = str, default = './models/0204_cbir_resnet50_2/cbirnet_epoch12_batchsize1.pth', help = 'the load path for trained models')
    parser.add_argument('--load_com_path', type = str, default = './cc_model/ccnet_0129_resnet50_best.pth', help = 'the load path for trained models')
    parser.add_argument('--net_type', type = str, default = 'ImgRetriever', help = 'Used for composition classification')
    # Network parameters
    parser.add_argument('--loss_type', type = str, default = 'comb', help = 'type of loss function')
    parser.add_argument('--backbone_type', type = str, default = 'resnet50', help = 'choose the backbone')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--kcm_scale', type = int, default = 0, help = 'the scale of kcm')
    # Dataset parameters
    parser.add_argument('--train_root', type = str, default = "./data/train", help = 'the base testing folder')
    parser.add_argument('--test_root', type = str, default = "./data/val", help = 'the base testing folder')
    parser.add_argument('--test_results_root', type = str, default = "./data/test/0204_cbir_resnet50_2_results", help = 'the base testing folder')
    parser.add_argument('--target_path', type = str, default = None, help = 'the target image path')
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
    outputs, count = Retriever(opt, net, com_net)
    print("Average Scores: ", outputs)
    print("Count: ", count)
