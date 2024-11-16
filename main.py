#Torch
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

#Utils
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import csv   
import os
import time
import argparse
from tqdm import tqdm
from utils import utils, dataset, probability_metrics
from models import lstm_attention
import pytorch_warmup as warmup

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='data/diginetica/', help='dataset directory path: data/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=60, help='hidden state size of gru module')
parser.add_argument('--heads', type=int, default=2, help='num of heads')
parser.add_argument('--embed_dim', type=int, default=50, help='the dimension of item embedding')
parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate') #lr * lr_dc
parser.add_argument('--lr_dc_step', type=int, default=45, help='the number of steps after which the learning rate decay') 
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--max_len', type=int, default=20, help='max length of sequence')
args = parser.parse_args()
print(args)

torch.manual_seed(522)
np.random.seed(522)
here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset_path.split('/')[-2] == 'diginetica':
    datasetname = 'diginetica'
    datasetname1 = 'diginetica'
    n_items = 43098
elif args.dataset_path.split('/')[-2] in 'yoochoose1_64':
    datasetname = 'yoochoose164'
    datasetname1 = 'yoochoose1_64'
    n_items = 37484
elif args.dataset_path.split('/')[-2] in 'yoochoose1_4':
    datasetname = 'yoochoose14'
    datasetname1 = 'yoochoose1_4'
    n_items = 37484
else:
    raise Exception('Unknown Dataset!')
    
MODEL_VARIATION = datasetname + "_LSTM_ATT_"

def main():
    print(f'Loading data for dataset {args.dataset_path} and model variation {MODEL_VARIATION}!')
    train, valid, test = dataset.load_data(args.dataset_path, valid_portion=args.valid_portion, maxlen=args.max_len)
    
    train_data = dataset.RecSysDataset(train)
    valid_data = dataset.RecSysDataset(valid)
    test_data = dataset.RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn=lambda data: utils.collate_fn(data, max_len=args.max_len))
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn=lambda data: utils.collate_fn(data, max_len=args.max_len))
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn=lambda data: utils.collate_fn(data, max_len=args.max_len))


    model = lstm_attention.LSTMAttentionModel(n_items, 
                                              args.hidden_size, 
                                              args.embed_dim, 
                                              args.batch_size,
                                              num_heads=args.heads).to(device) 

    optimizer = optim.Adam(params=model.parameters(), 
                           lr=args.lr)  
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999)) #, weight_decay=0.01
    # num_steps = len(train_loader) * args.epoch
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    
    criterion = nn.CrossEntropyLoss()

    # scheduler = StepLR(optimizer, 
    #                    step_size = args.lr_dc_step, 
    #                    gamma = args.lr_dc)
    
    early_stopper = utils.EarlyStopper(patience=5, 
                                       min_delta=10)

    # Info
    losses, valid_losses = [], []
    valid_recall, valid_mrr, valid_hit = 0,0,0
    best_recall, best_mrr, best_hit, best_epoch = 0,0,0,0
    best_valid_loss = float('inf')
    now = datetime.now()
    now_time = time.time()
    timestamp = now.strftime("%d_%m_%Y_%H:%M:%S")

    for epoch in tqdm(range(args.epoch)):
        # Train  warmup_scheduler, lr_scheduler,
        epoch_loss = trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 200)
        losses.append(epoch_loss)

        # Validation       
        valid_recall, valid_mrr, valid_hit, valid_loss = validate(valid_loader, model, criterion)
        valid_losses.append(valid_loss)
        print(f"Epoch {epoch} validation: Recall@20: {valid_recall:.4f}, MRR@20: {valid_mrr:.4f}, HIT@20: {valid_hit:.4f}, Validation loss: {valid_loss:.4f} \n")

        # Checkpoint
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_recall, best_mrr, best_hit, best_epoch = valid_recall, valid_mrr, valid_hit, epoch
            ckpt_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(ckpt_dict, 'checkpoints/'+MODEL_VARIATION+'latest_checkpoint_'+timestamp+'.pth.tar')
            torch.save(model.embedding.weight.data, 'embeddings/'+MODEL_VARIATION+'latest_checkpoint_'+timestamp+'.pth.tar')        

        # Patience
        if early_stopper.early_stop(valid_loss):    
            print(f"Early stop in epoch {epoch}!")         
            break

    # Plot losses
    print('--------------------------------')
    print('Plotting loss curve...')
    plt.clf()
    plt.plot(losses[1:],  label='Training Loss')
    plt.plot(valid_losses[1:], label='Validation Loss')
    plt.title('Training/Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')  
    plt.savefig('loss_curves/'+MODEL_VARIATION+'loss_curve_'+timestamp+'.png')

    # Test model
    ckpt = torch.load('checkpoints/'+MODEL_VARIATION+'latest_checkpoint_'+timestamp+'.pth.tar')
    model.load_state_dict(ckpt['state_dict'])
    test_recall, test_mrr, test_hit, _ = validate(test_loader, model, criterion)
    print(f"Test: Recall@20: {test_recall:.4f}, MRR@20: {test_mrr:.4f}, HIT@20: {test_hit:.4f}, Best Epoch: {ckpt['epoch']}")

    # Save metrics
    model_unique_id = MODEL_VARIATION + timestamp
    fields=[model_unique_id, test_recall, test_mrr, test_hit,timestamp,(time.time() - now_time),valid_recall, valid_mrr, valid_hit, args.lr, args.hidden_size, args.batch_size, args.embed_dim, datasetname, args.epoch, args.topk, args.max_len, best_recall, best_mrr, best_hit, best_epoch]  
    with open(r'stats/data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion,  log_aggr=1): #warmup_scheduler, lr_scheduler,
    model.train()
    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)

        if (torch.any(torch.isnan(seq)))|(torch.any(torch.isnan(target))):
            print("NaN values found in", epoch, i)
            break
        
        optimizer.zero_grad()
        outputs = model(seq, lens)

        loss = criterion(outputs, target)
        loss.backward()

        if (torch.any(torch.isnan(seq)))|(torch.any(torch.isnan(target)))|(torch.any(torch.isnan(loss))):
            print("NaN values found in", epoch, i)
            break

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step() 
        # with warmup_scheduler.dampening():
        #     #print('stepping')
        #     lr_scheduler.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val
        #iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print(f"[TRAIN] epoch {epoch + 1}/{num_epochs} batch loss: {loss_val:.4f} (avg {sum_epoch_loss / (i + 1):.4f}) ({len(seq) / (time.time() - start):.2f} im/s)")

        start = time.time()
    
    epoch_loss = sum_epoch_loss/len(train_loader)
    return epoch_loss


def validate(valid_loader, model, criterion):
    model.eval()
    sum_valid_loss = 0
    recalls, mrrs, hits = [], [], []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)

            # Validation loss
            loss = criterion(outputs, target)
            sum_valid_loss += loss.item()

            # Metrics
            logits = F.softmax(outputs, dim = 1)
            recall, mrr, hit = probability_metrics.evaluate(logits, target, k = args.topk)
            recalls.append(recall)
            mrrs.append(mrr)
            hits.append(hit)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    mean_hit = np.mean(hit)
    loss = sum_valid_loss/len(valid_loader)
    return mean_recall, mean_mrr, mean_hit, loss

if __name__ == '__main__':
    main()