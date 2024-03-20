import os
import os.path as osp
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import network_cc
import network
# from network_vgg import VGGFeaNet

# ----------------------------------------
#                Dataset
# ----------------------------------------

def get_files(path):
    # read a folder, return the complete path
    imgs_folder = osp.join(path, 'imgs')
    img_files = os.listdir(imgs_folder)
    img_files.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])))
    img_paths = [os.path.join(imgs_folder, img_file) for img_file in img_files]
    return img_paths

def get_test_files(path):
    # read a folder, return the complete path
    imgs_folder = osp.join(path, 'imgs')
    img_files = os.listdir(imgs_folder)
    img_files.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])))
    tmp_dict = {}
    useful_list = []
    for img_name in img_files:
        prefix =  img_name.split('_')[0] + '_' + img_name.split('_')[1]
        if prefix not in tmp_dict:
            tmp_dict[prefix] = 1
        else:
            tmp_dict[prefix] += 1
    # print(tmp_dict)
    for key in tmp_dict:
        if tmp_dict[key] > 2:
            center_name = f"{key}_{tmp_dict[key] // 2}.jpg"
            useful_list.append(center_name)
        
    img_paths = [os.path.join(imgs_folder, useful_name) for useful_name in useful_list]
    return img_paths

def get_single_test_files(path):
    # read a folder, return the complete path
    imgs_folder = osp.join(path, 'imgs')
    img_files = os.listdir(imgs_folder)
    img_paths = [os.path.join(imgs_folder, name) for name in img_files]
    return img_paths

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

def create_net(opt, is_com = False):
    # Initialize the networks
    if is_com:
        net = getattr(network_cc, 'ComClassifier')(opt)
        print('com net is created!')
        pretrained_net = torch.load(opt.load_com_path)
        net = load_dict(net, pretrained_net)
        print('Load the com net with %s' % opt.load_com_path)
    else:
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

def cosine_distance(x1, x2):
    cos_sim = F.cosine_similarity(x1, x2, dim=-1)
    cos_distance = 1.0 - cos_sim
    return cos_distance

def save_model(net, epoch, opt, best = False):
    if best:
        model_name = 'cbirnet_best.pth'
    else:
        model_name = 'cbirnet_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
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
            
            
def visualize_ir_preds(opt, img, kcm, epoch):
    # print(img.shape)
    img = (img * opt.norm_std) + opt.norm_mean
    img = img.squeeze().numpy()
    img = np.clip((img * 255), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dst = img.copy()
    
    kcm = kcm.permute(0, 2, 3, 1)[0].detach().cpu().numpy().astype(np.float32)
    # norm_kcm = np.zeros((height, width, 1))
    norm_kcm = cv2.normalize(kcm, None, 0, 255, cv2.NORM_MINMAX)
    norm_kcm = np.asarray(norm_kcm, dtype=np.uint8)
    heat_im = cv2.applyColorMap(norm_kcm, cv2.COLORMAP_JET)
    # heat_im = cv2.cvtColor(heat_im, cv2.COLOR_BGR2RGB)
    
    heat_im = cv2.resize(heat_im, (opt.size_w, opt.size_h))
    fuse_im = cv2.addWeighted(img, 0.2, heat_im, 0.8, 0, dtype=cv2.CV_8U)
    fuse_im = np.concatenate([dst, fuse_im], axis=1)
    check_path(opt.sample_val_path)
    img_name = 'val_sample_epoch%d.png' % epoch
    cv2.imwrite(os.path.join(opt.sample_val_path, img_name), fuse_im)
