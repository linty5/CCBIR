import argparse
import trainer

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--save_path', type = str, default = './models/model_name', help = 'the save path for trained models')
    parser.add_argument('--sample_train_path', type = str, default = './samples', help = 'the save path for training samples')
    parser.add_argument('--sample_val_path', type = str, default = './samples_val/model_name', help = 'the save path for validation samples')
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'nn.Parallel needs or not')
    parser.add_argument('--net_type', type = str, default = 'ImgRetriever', help = 'Used for composition classification')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    parser.add_argument('--evaluate_interval', type = int, default = 1, help = 'interval between evaluations on validation set')
    parser.add_argument('--load_com_path', type = str, default = './cc_model/ccnet_best.pth', help = 'the load path of com models')
    parser.add_argument('--load_path', type = str, default = None, help = 'the load path for trained models')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 12, help = 'number of epochs of training')
    parser.add_argument('--epochs_overhead', type = int, default = 0, help = 'number of trained epochs')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--ac_batch_size', type = int, default = 8, help = 'size of the batches')
    parser.add_argument('--grad_accumulation', type = bool, default = True, help = 'accumulate the gradient')
    parser.add_argument('--loss_type', type = str, default = 'cosine', help = 'type of loss function')
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_l1', type = float, default = 1, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_l2', type = float, default = 1, help = 'the parameter of L2Loss')
    parser.add_argument('--lambda_neg', type = float, default = 1, help = 'the parameter of negative samples')
    parser.add_argument('--lambda_triplet', type = float, default = 1, help = 'the parameter of tripletloss')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--backbone_type', type = str, default = 'resnet50', help = 'choose the backbone')
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--kcm_scale', type = int, default = 0.5, help = 'the scale of kcm')
    # Dataset parameters
    parser.add_argument('--train_root', type = str, default = "./data/train", help = 'the base training folder')
    parser.add_argument('--val_root', type = str, default = "./data/val", help = 'the base validating folder')
    parser.add_argument('--test_root', type = str, default = "./data/val", help = 'the base testing folder')
    parser.add_argument('--comp_types_num', type = int, default = 9, help = 'composition types num')
    parser.add_argument('--norm_mean', type = float, default = 0.400, help = 'set None to recalculate')
    parser.add_argument('--norm_std', type = float, default = 0.147, help = 'set None to recalculate')
    parser.add_argument('--size_w', type = int, default = 256, help = 'size of image')
    parser.add_argument('--size_h', type = int, default = 256, help = 'size of image')
    opt = parser.parse_args()
    print(opt)
    
    # Enter main function
    trainer.Trainer(opt)
    