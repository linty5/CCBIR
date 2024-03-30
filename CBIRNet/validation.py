import argparse
import torch.nn as nn
import tqdm
import torch
from torch.utils.data import DataLoader

import dataset
import utils


def Evaluator(opt, net, com_net, epoch = None):
    net.eval()
    net = net.cuda()
    com_net.eval()
    com_net = com_net.cuda()
    
    utils.check_path(opt.sample_val_path)
    
    cos = torch.nn.CosineSimilarity()
    def neg_CosineSimilarity(a,b):
        return -1*cos(a,b)
    criterion_cos = nn.CosineEmbeddingLoss(reduction='sum')
    criterion_sim =  torch.nn.TripletMarginWithDistanceLoss(distance_function=neg_CosineSimilarity, reduction='sum')
    criteriopn_l1 = nn.L1Loss()
    criteriopn_l2 = nn.MSELoss()
    count = 0
    error_sum = 0

    with torch.no_grad():
        temp_img_origin = 0
        temp_img_pos = 0
        temp_img_neg = 0
        
        val_dataset = dataset.RetrievalValDataset(opt)
        print('The overall number of evaluating images equals to %d' % len(val_dataset))
        val_loader = DataLoader(val_dataset, 
                                batch_size = 1, 
                                shuffle = False, 
                                num_workers = 1)
        
        for idx, (img_origin, img_pos, img_neg) in enumerate(tqdm.tqdm(val_loader)):
            if img_origin.numpy().shape == (1,):
                print("Skipping batch due to empty batch")
                continue
            
            temp_img_origin = img_origin
            temp_img_pos = img_pos
            temp_img_neg = img_neg
            # Load and put to cuda
            img_origin = img_origin.cuda()                                          # out: [B, 1, 256, 256]
            img_pos = img_pos.cuda()                                                # out: [B, 1, 256, 256]
            img_neg = img_neg.cuda()                                                # out: [B, 1, 256, 256]

            batch_length = img_origin.shape[0]
            
            _, kcm_origin = com_net(img_origin)
            _, kcm_pos = com_net(img_pos)
            _, kcm_neg = com_net(img_neg)

            # forward propagation
            out_img_origin = net(img_origin, kcm_origin)
            out_img_pos = net(img_pos, kcm_pos)
            out_img_neg = net(img_neg, kcm_neg)
            
            # Compute losses
            if opt.loss_type == "triplet":
                error = criterion_sim(out_img_origin, out_img_pos, out_img_neg)
            elif opt.loss_type == "cosine":
                c1 = criterion_cos(out_img_origin, out_img_pos, torch.ones(batch_length).cuda())
                c2 = criterion_cos(out_img_origin, out_img_neg, -torch.ones(batch_length).cuda())
                error = c1 + c2
            else:
                sim = criterion_sim(out_img_origin, out_img_pos, out_img_neg)
                l1 = torch.clamp(criteriopn_l1(out_img_origin, out_img_pos) + opt.lambda_neg * (torch.exp(criteriopn_l1(out_img_origin, out_img_neg)) - 1) + opt.lambda_neg * (torch.exp(criteriopn_l1(out_img_pos, out_img_neg)) - 1), min=0, max=5)
                l2 = torch.clamp(criteriopn_l2(out_img_origin, out_img_pos) + opt.lambda_neg * (torch.exp(criteriopn_l2( out_img_origin, out_img_neg)) - 1) + opt.lambda_neg * (torch.exp(criteriopn_l2(out_img_pos, out_img_neg)) - 1), min=0, max=5)
                error = opt.lambda_triplet * sim + opt.lambda_l1 * l1 + opt.lambda_l2 * l2
            error_sum += error
            count += 1
            
    utils.visualize_ir_preds(opt, temp_img_origin.cpu(), kcm_origin, epoch)
    utils.visualize_ir_preds(opt, temp_img_pos.cpu(), kcm_pos, epoch)
    utils.visualize_ir_preds(opt, temp_img_neg.cpu(), kcm_neg, epoch)
    
    # print(count)

    return error_sum/count
    
if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--load_path', type = str, default = './models/model_name/cbirnet_best.pth', help = 'the load path for trained models')
    parser.add_argument('--load_com_path', type = str, default = './cc_model/ccnet_best_resnet50.pth', help = 'the load path for trained models')
    parser.add_argument('--net_type', type = str, default = 'ImgRetriever', help = 'Used for composition classification')
    parser.add_argument('--sample_val_path', type = str, default = './samples_val', help = 'the save path for validation samples')
    parser.add_argument('--lambda_l1', type = float, default = 1, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_l2', type = float, default = 1, help = 'the parameter of L2Loss')
    parser.add_argument('--lambda_neg', type = float, default = 1, help = 'the parameter of negative samples')
    parser.add_argument('--lambda_triplet', type = float, default = 1, help = 'the parameter of tripletloss')
    # Network parameters
    parser.add_argument('--loss_type', type = str, default = 'cosine', help = 'type of loss function')
    parser.add_argument('--backbone_type', type = str, default = 'resnet50', help = 'choose the backbone')
    parser.add_argument('--start_channels', type = int, default = 128, help = 'latent channels')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--kcm_scale', type = int, default = 0.5, help = 'the scale of kcm')
    # Dataset parameters
    parser.add_argument('--train_root', type = str, default = "./data/train", help = 'the base validating folder')
    parser.add_argument('--val_root', type = str, default = "./data/val", help = 'the base validating folder')
    parser.add_argument('--test_root', type = str, default = "./data/val", help = 'the base testing folder')
    parser.add_argument('--comp_types_num', type = int, default = 9, help = 'composition types num')
    parser.add_argument('--norm_mean', type = float, default = 0.400, help = 'set None to recalculate')
    parser.add_argument('--norm_std', type = float, default = 0.147, help = 'set None to recalculate')
    parser.add_argument('--size_w', type = int, default = 256, help = 'size of image')
    parser.add_argument('--size_h', type = int, default = 256, help = 'size of image')
    parser.add_argument('--mode', type = str, default = "val", help = 'choose val or test')
    opt = parser.parse_args()
    print(opt)
    
    net = utils.create_net(opt)
    net = net.cuda()
    com_net = utils.create_net(opt, is_com = True)
    com_net = com_net.cuda()
    error = Evaluator(opt, net, com_net, 0)
    print(error)