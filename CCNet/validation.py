import argparse
from sklearn.metrics import precision_recall_fscore_support
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import utils

def Evaluator(opt, net, comp_types, epoch = 0):
    net.eval()
    net = net.cuda()
    
    total = 0
    correct = 0
    cls_cnt = [0 for i in range(9)]
    cls_correct = [0 for i in range(9)]
    
    utils.check_path(opt.sample_val_path)
    
    y_true = []
    y_pred = []

    with torch.no_grad():
        
        val_dataset = dataset.CompositionValDataset(opt)
        print('The overall number of evaluating images equals to %d' % len(val_dataset))
        val_loader = DataLoader(val_dataset, 
                                batch_size = 1, 
                                shuffle = False, 
                                num_workers = 1)
        
        for idx, (img, label) in enumerate(tqdm.tqdm(val_loader)):
            img = img.cuda() 
            preds, kcm = net(img)
            preds = preds.cpu()
            _,predicted = torch.max(preds.data,1)
            
            total += label.shape[0]
            pr = predicted[0].item()
            gt = label[0].numpy().astype(int)
            gt = np.where(gt == 1)[0]
            # print(pr, "   ",  gt)
            if len(gt) == 0:
                continue
            y_pred.append(pr)
            if pr in gt:
                correct += 1
                cls_cnt[pr] += 1
                cls_correct[pr] += 1
                y_true.append(pr)
            else:
                cls_cnt[gt[0]] += 1
                y_true.append(gt[0])
            # print(y_pred)
            # print(y_true)
            
            if idx % 100 == 0:
                utils.visualize_com_preds(opt, img.cpu(), preds, kcm, label, opt.sample_val_path, comp_types, idx, epoch)
    acc = float(correct) / total
    utils.visualize_com_preds(opt, img.cpu(), preds, kcm, label, opt.sample_val_path, comp_types, idx, epoch)
    print('Test on {} images, {} Correct, Acc {:.2%}'.format(total, correct, acc))
    for i in range(len(cls_cnt)):
        if cls_cnt[i] == 0:
            print('{}: total {} images, {} correct, Acc {:.2%}'.format(
                comp_types[i], cls_cnt[i], cls_correct[i], 0))
        else:
            print('{}: total {} images, {} correct, Acc {:.2%}'.format(
                comp_types[i], cls_cnt[i], cls_correct[i], float(cls_correct[i]) / cls_cnt[i]))
            
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    print('Precision: {:.2%}'.format(precision))
    print('Recall: {:.2%}'.format(recall))
    print('F1 Score: {:.2%}'.format(f1))

    return acc, f1, precision, recall
    
if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--load_path', type = str, default = './models/model_name/ccnet_best.pth', help = 'the load path for trained models')
    parser.add_argument('--net_type', type = str, default = 'ComClassifier', help = 'Used for composition classification')
    parser.add_argument('--sample_val_path', type = str, default = './samples_val', help = 'the save path for validation samples')
    # Network parameters
    parser.add_argument('--backbone_type', type = str, default = 'resnet50', help = 'choose the backbone')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    # Dataset parameters
    parser.add_argument('--val_root', type = str, default = "./data/val", help = 'the base validating folder')
    parser.add_argument('--test_root', type = str, default = "./data/selected_t", help = 'the base testing folder')
    parser.add_argument('--comp_types_num', type = int, default = 9, help = 'composition types num')
    parser.add_argument('--norm_mean', type = float, default = 0.457, help = 'set None to recalculate')
    parser.add_argument('--norm_std', type = float, default = 0.210, help = 'set None to recalculate')
    parser.add_argument('--size_w', type = int, default = 256, help = 'size of image')
    parser.add_argument('--size_h', type = int, default = 256, help = 'size of image')
    opt = parser.parse_args()
    print(opt)
    
    comp_types = ['rule of thirds(RoT)', 'vertical', 'horizontal', 'diagonal', 
                  'curved', 'triangle', 'center', 'symmetric', 'pattern']

    net = utils.create_net(opt)
    net = net.cuda()
    acc, f1, precision, recall = Evaluator(opt, net, comp_types)
