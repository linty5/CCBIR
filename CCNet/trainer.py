import time
import datetime
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import dataset
import utils
import validation

def Trainer(opt, comp_types):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------
    wandb.init(
        project='CCBIR-CCNet',
        name = opt.save_path.split('/')[-1],
        config=opt)

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    if opt.multi_gpu == True:
        opt.batch_size *= gpu_num
        opt.num_workers *= gpu_num
        print("Batch size is changed to %d" % opt.batch_size)
        print("Number of workers is changed to %d" % opt.num_workers)
    
    # Build networks
    net = utils.create_net(opt)

    # To device
    if opt.multi_gpu == True:
        net = nn.DataParallel(net)
        net = net.cuda()
    else:
        net = net.cuda()

    # Loss functions
    criterion_CE = torch.nn.CrossEntropyLoss()

    # Optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=36, eta_min=1e-7)

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    train_dataset = dataset.CompositionTrainDataset(opt)
    print('The overall number of training images equals to %d' % len(train_dataset))


    # Define the dataloader
    train_loader = DataLoader(train_dataset, 
                              batch_size = opt.batch_size, 
                              shuffle = False, 
                              num_workers = opt.num_workers, 
                              pin_memory = True)
    
    utils.check_path(opt.sample_train_path)
    
    # ----------------------------------------
    #                Training
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()
    
    best_acc = -1
    best_epoch = 0

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (img, label) in enumerate(train_loader):

            # Load and put to cuda
            img = img.cuda()                                                # out: [B, 1, 256, 256]
            optimizer.zero_grad()

            # forward propagation
            preds, _ = net(img)
            
            # Compute losses
            label = label.cuda()  
            if label.dim() == 2:
                label = label.squeeze(1)
            Loss_CE = criterion_CE(preds, label)
            loss = Loss_CE
            loss.backward()
            optimizer.step()

            wandb.log({'Loss_CE': Loss_CE.item(), 'LearningRate': optimizer.param_groups[0]['lr']}) 
                       
            # Determine approximate time left
            batches_done = epoch * len(train_loader) + batch_idx
            batches_left = opt.epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [CE Loss: %.5f] time_left: %s" %
                ((epoch + opt.epochs_overhead + 1), opt.epochs, batch_idx, len(train_loader), loss.item(), time_left))
            
        # Save the model
        utils.save_model(net, (epoch + opt.epochs_overhead + 1), opt)
        scheduler.step()
        
        if (epoch+1) % opt.evaluate_interval == 0:
            acc, f1, precision, recall = validation.Evaluator(opt, net, comp_types, epoch)
            wandb.log({'Val Accuracy': acc})
            wandb.log({'Val F1': f1})
            wandb.log({'Val Precision': precision})
            wandb.log({'Val Recall': recall})
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                utils.save_model(net, (epoch + opt.epochs_overhead + 1), opt, best = True)
            wandb.log({'Best Accuracy': best_acc})
            wandb.log({'Best Epoch': best_epoch})
                
                    
    wandb.finish()
        
