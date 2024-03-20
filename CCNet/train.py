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
    parser.add_argument('--net_type', type = str, default = 'ComClassifier', help = 'Used for composition classification')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    parser.add_argument('--evaluate_interval', type = int, default = 1, help = 'interval between evaluations on validation set')
    parser.add_argument('--load_path', type = str, default = None, help = 'the load path of models')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 12, help = 'number of epochs of training')
    parser.add_argument('--epochs_overhead', type = int, default = 0, help = 'number of trained epochs')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'size of the batches')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--backbone_type', type = str, default = 'resnet50', help = 'choose the backbone')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
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
    
    comp_types = ['rule of thirds(RoT)', 'vertical', 'horizontal', 'diagonal', 
                  'curved', 'triangle', 'center', 'symmetric', 'pattern']
    
    # Enter main function
    trainer.Trainer(opt, comp_types)
    