import os
import os.path as osp
import numpy as np
import cv2
import skimage.measure
import torch
import torch.nn as nn

import network as network
# from network_vgg import VGGFeaNet

# ----------------------------------------
#                Dataset
# ----------------------------------------

def get_files(path, is_train=True):
    # read a folder, return the complete path
    imgs_folder = osp.join(path, 'imgs')
    img_names_sorted = sorted(os.listdir(imgs_folder), key=lambda k: float(os.path.splitext(k)[0]))
    img_paths = [os.path.join(imgs_folder, img_file) for img_file in img_names_sorted]
    if is_train:
        label_path = osp.join(path, 'label.txt')
        with open(label_path, 'r') as file:
            lines = file.readlines()
        labels = []
        for line in lines:
            label = [float(x) for x in line.strip().split()]
            labels.append(torch.tensor(label))
        if len(labels) != len(img_paths):
            raise ValueError('The number of image files and labels is not equal.' + \
                             'image_paths: %d. labels: %d' % (len(img_paths), len(labels)))
        return img_paths, labels
    else:
        return img_paths, []

def cal_norm_paras(img_paths):
    mean_accumulator = 0.0
    std_accumulator = 0.0
    
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mean = np.mean(img)
        std = np.std(img)
        mean_accumulator += mean
        std_accumulator += std

    total_mean = mean_accumulator / len(img_paths)
    total_std = std_accumulator / len(img_paths)
    
    print('The mean of the dataset is %f' % total_mean, 'The std of the dataset is %f' % total_std)

    return total_mean, total_std
    
# ----------------------------------------
#                Networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
            
    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def create_net(opt):
    # Initialize the networks
    net = getattr(network, opt.net_type)(opt)
    weights_init(net, init_type = 'normal', init_gain = 0.02)
    print('net is created!')
    # Init the networks
    if opt.load_path:
        pretrained_net = torch.load(opt.load_path)
        net = load_dict(net, pretrained_net)
        print('Load the net with %s' % opt.load_path)
    return net

def load_dict(process_net, pretrained_net):
    pretrained_dict = pretrained_net
    process_dict = process_net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    process_dict.update(pretrained_dict)
    process_net.load_state_dict(process_dict)
    return process_net

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ----------------------------------------
#    Training and Validation
# ----------------------------------------

def save_model(net, epoch, opt, best = False):
    if best:
        model_name = 'ccnet_best.pth'
    else:
        model_name = 'ccnet_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
    check_path(opt.save_path)
    model_path = os.path.join(opt.save_path, model_name)
    if opt.multi_gpu == True:
        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.module.state_dict(), model_path)
            print('The trained model is successfully saved at epoch %d' % (epoch))
    else:
        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), model_path)
            print('The trained model is successfully saved at epoch %d' % (epoch))
            
            
def visualize_com_preds(opt, img, preds, kcm, label, sample_val_path, comp_types, idx, epoch = 0):
    _, predicted = torch.max(preds.data, 1)
    # print('Composition prediction', predicted)
    # print('Ground-truth composition', category)
    predicted_name = comp_types[predicted[0].item()]
    if label is not None:
        gt_label = [comp_types[int(c)] for c in label[0].tolist()]
    else:
        gt_label = 'None'
    img = (img * opt.norm_std) + opt.norm_mean
    img = img.squeeze().numpy()
    img = np.clip((img * 255), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dst = img.copy()
    gt_ss = 'gt:{}'.format(gt_label)
    dst = cv2.putText(dst, gt_ss, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    pr_ss = 'predict:{}'.format(predicted_name)
    dst = cv2.putText(dst, pr_ss, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    # h,w,1
    kcm = kcm.permute(0, 2, 3, 1)[0].detach().cpu().numpy().astype(np.float32)
    # norm_kcm = np.zeros((height, width, 1))
    norm_kcm = cv2.normalize(kcm, None, 0, 255, cv2.NORM_MINMAX)
    norm_kcm = np.asarray(norm_kcm, dtype=np.uint8)
    heat_im = cv2.applyColorMap(norm_kcm, cv2.COLORMAP_JET)
    # heat_im = cv2.cvtColor(heat_im, cv2.COLOR_BGR2RGB)
    heat_im = cv2.resize(heat_im, (opt.size_w, opt.size_h))
    fuse_im = cv2.addWeighted(img, 0.2, heat_im, 0.8, 0, dtype=cv2.CV_8U)
    # fuse_im = np.concatenate([dst, fuse_im], axis=1)
    check_path(sample_val_path)
    
    img_name = 'val_sample_epoch%d_idx%d_img.jpg' % (epoch, idx)
    cv2.imwrite(os.path.join(sample_val_path, img_name), img)
    dst_name = 'val_sample_epoch%d_idx%d_dst.jpg' % (epoch, idx)
    cv2.imwrite(os.path.join(sample_val_path, dst_name), dst)
    heat_name = 'val_sample_epoch%d_idx%d_heat.jpg' % (epoch, idx)
    cv2.imwrite(os.path.join(sample_val_path, heat_name), heat_im)
    fuse_name = 'val_sample_epoch%d_idx%d_fuse.jpg' % (epoch, idx)
    cv2.imwrite(os.path.join(sample_val_path, fuse_name), fuse_im)
