import time
import datetime
import sys
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import dataset
import utils
import validation

def Trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------
    wandb.init(
        project='CCBIR-CBIRNet', 
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
    com_net = utils.create_net(opt, is_com = True)
    com_net.eval()

    # To device
    if opt.multi_gpu == True:
        net = nn.DataParallel(net)
        net = net.cuda()
        com_net = nn.DataParallel(com_net)
        com_net = com_net.cuda()
    else:
        net = net.cuda()
        com_net = com_net.cuda()

    # Loss functions
    criterion_cos = nn.CosineEmbeddingLoss(reduction='mean')
    criterion_sim =  torch.nn.TripletMarginWithDistanceLoss(distance_function=utils.cosine_distance, reduction='mean')
    criteriopn_l1 = nn.L1Loss()
    criteriopn_l2 = nn.MSELoss()
    
    # Optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    if opt.grad_accumulation:
        accumulation_steps = opt.ac_batch_size
    else:
        accumulation_steps = 1
    accumulated_gradients = {name: torch.zeros_like(param) for name, param in net.named_parameters()}

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=1e-6)

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    train_dataset = dataset.RetrievalTrainDataset(opt)
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
    
    best_val_loss = sys.float_info.max
    best_epoch = 0

    # Training loop
    for epoch in range(opt.epochs):
        count = 0
        for batch_idx, (img_origin, img_pos, img_neg) in enumerate(train_loader):
            if img_origin.numpy().shape == (1,):
                print("Skipping batch due to empty batch")
                continue
            
            batch_length = img_origin.shape[0]
            
            # Load and put to cuda
            img_origin = img_origin.cuda()                                          # out: [B, 1, 256, 256]
            img_pos = img_pos.cuda()                                                # out: [B, 1, 256, 256]
            img_neg = img_neg.cuda()                                                # out: [B, 1, 256, 256]
            optimizer.zero_grad()
            
            with torch.no_grad():
                _, kcm_origin = com_net(img_origin)
                _, kcm_pos = com_net(img_pos)
                _, kcm_neg = com_net(img_neg)
            torch.enable_grad()

            # forward propagation
            out_img_origin = net(img_origin, kcm_origin)
            out_img_pos = net(img_pos, kcm_pos)
            out_img_neg = net(img_neg, kcm_neg)
            
            # Compute losses
            if opt.loss_type == "triplet":
                loss = criterion_sim(out_img_origin, out_img_pos, out_img_neg)
                wandb.log({'Train\BatchLoss': loss.item() / batch_length})
            elif opt.loss_type == "cosine":
                c1 = criterion_cos(out_img_origin, out_img_pos, torch.ones(batch_length).cuda())
                c2 = criterion_cos(out_img_origin, out_img_neg, -torch.ones(batch_length).cuda())
                loss = c1 + c2
                wandb.log({'Train\BatchLoss': loss.item() / batch_length, "Train\BatchPosSampleError": c1, "Train\BatchNegSampleError": c2})
            else:
                sim = criterion_sim(out_img_origin, out_img_pos, out_img_neg)
                l1 = torch.clamp(criteriopn_l1(out_img_origin, out_img_pos) + opt.lambda_neg * (torch.exp(criteriopn_l1(out_img_origin, out_img_neg)) - 1) + opt.lambda_neg * (torch.exp(criteriopn_l1(out_img_pos, out_img_neg)) - 1), min=0, max=5)
                l2 = torch.clamp(criteriopn_l2(out_img_origin, out_img_pos) + opt.lambda_neg * (torch.exp(criteriopn_l2( out_img_origin, out_img_neg)) - 1) + opt.lambda_neg * (torch.exp(criteriopn_l2(out_img_pos, out_img_neg)) - 1), min=0, max=5)
                loss = opt.lambda_triplet * sim + opt.lambda_l1 * l1 + opt.lambda_l2 * l2
                wandb.log({'Train\BatchLoss': loss.item() / batch_length, "Train\BatchSimLoss": sim, "Train\BatchL1Loss": l1, "Train\BatchL2Loss": l2})
            
            loss.backward()
            
            for name, param in net.named_parameters():
                accumulated_gradients[name] += param.grad
                
            if (count + 1) % accumulation_steps == 0:
                for name, param in net.named_parameters():
                    param.grad = accumulated_gradients[name] / accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
                accumulated_gradients = {name: torch.zeros_like(param) for name, param in net.named_parameters()}

            wandb.log({'LearningRate': optimizer.param_groups[0]['lr']}) 
                       
            # Determine approximate time left
            batches_done = epoch * len(train_loader) + batch_idx
            batches_left = opt.epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.5f] time_left: %s" %
                ((epoch + opt.epochs_overhead + 1), opt.epochs, batch_idx, len(train_loader), loss.item(), time_left))

            count += 1
            
        # Save the model
        utils.save_model(net, (epoch + opt.epochs_overhead + 1), opt)
        scheduler.step()
        
        if (epoch+1) % opt.evaluate_interval == 0:
            val_loss = validation.Evaluator(opt, net, com_net, epoch)
            wandb.log({'Val Loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                utils.save_model(net, (epoch + opt.epochs_overhead + 1), opt, best = True)
            wandb.log({'Best Val Loss': best_val_loss})
            wandb.log({'Best Epoch': best_epoch})
                
    wandb.finish()