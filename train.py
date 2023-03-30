import os
import math
import numpy as np
import pandas as pd
import json
import operator
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
import wandb

from data import create_dataloader
from model import PharmHGT as Model
from schedular import NoamLR
from utils import get_func,remove_nan_label

def evaluate(dataloader,model,device,metric_fn,metric_dtype,task):
    metric = 0
    for bg,labels in dataloader:
        bg,labels = bg.to(device),labels.type(metric_dtype)
        pred = model(bg).cpu().detach()
        if task == 'classification':
            pred = torch.sigmoid(pred)
        elif task == 'multiclass':
            pred = torch.softmax(pred,dim=1)
        num_task =  pred.size(1)
        if num_task >1:
            m = 0
            for i in range(num_task):
                try:
                    m += metric_fn(*remove_nan_label(pred[:,i],labels[:,i]))
                except:
                    print(f'only one class for task {i}')
            m = m/num_task
        else:
            m = metric_fn(pred,labels.reshape(pred.shape))
        metric += m.item()*len(labels)
    metric = metric/len(dataloader.dataset)
    
    return metric

def train(data_args,train_args,model_args,seeds=[0,100,200,300,400]):
    
    epochs = train_args['epochs']
    device = train_args['device'] if torch.cuda.is_available() else 'cpu'
    save_path = train_args['save_path']

    wandb.config = train_args

    os.makedirs(save_path,exist_ok=True)
    
    
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        for fold in range(train_args['num_fold']):
            wandb.init(project='PharmHGT', entity='entity_name',group=train_args["data_name"],name=f'seed{seed}_fold{fold}',reinit=True)
            trainloader = create_dataloader(data_args,f'{seed}_fold_{fold}_train.csv',shuffle=True)
            valloader = create_dataloader(data_args,f'{seed}_fold_{fold}_valid.csv',shuffle=False,train=False)
            testloader = create_dataloader(data_args,f'{seed}_fold_{fold}_test.csv',shuffle=False,train=False)
            print(f'dataset size, train: {len(trainloader.dataset)}, \
                    val: {len(valloader.dataset)}, \
                    test: {len(testloader.dataset)}')
            model = Model(model_args).to(device)
            optimizer = Adam(model.parameters())
            scheduler = NoamLR(
                optimizer=optimizer,
                warmup_epochs=[train_args['warmup']],
                total_epochs=[epochs],
                steps_per_epoch=len(trainloader.dataset) // data_args['batch_size'],
                init_lr=[train_args['init_lr']],
                max_lr=[train_args['max_lr']],
                final_lr=[train_args['final_lr']]
            )

            loss_fn = get_func(train_args['loss_fn'])
            metric_fn = get_func(train_args['metric_fn'])
            if train_args['loss_fn'] in []:
                loss_dtype = torch.long
            else:
                loss_dtype = torch.float32

            if train_args['metric_fn'] in []:
                metric_dtype = torch.long
            else:
                metric_dtype = torch.float32

            if train_args['metric_fn'] in ['auc','acc']:
                best = 0
                op = operator.ge
            else:
                best = np.inf
                op = operator.le
            best_epoch = 0
            
            for epoch in tqdm(range(epochs)):
                model.train()
                total_loss = 0
                for bg,labels in trainloader:
                    bg,labels = bg.to(device),labels.type(loss_dtype).to(device)
                    pred = model(bg)
                    num_task =  pred.size(1)
                    if num_task > 1:
                        loss = 0
                        for i in range(num_task):
                            loss += loss_fn(*remove_nan_label(pred[:,i],labels[:,i]))
                    else:
                        loss = loss_fn(*remove_nan_label(pred,labels.reshape(pred.shape)))
                    total_loss += loss.item()*len(labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                total_loss = total_loss / len(trainloader.dataset)
                
                # val
                model.eval()
                val_metric = evaluate(valloader,model,device,metric_fn,metric_dtype,data_args['task'])
                if op(val_metric,best):
                    best = val_metric
                    best_epoch = epoch
                    torch.save(model.state_dict(),os.path.join(save_path,f'./best_fold{fold}.pt'))


                wandb.log({f'train {train_args["loss_fn"]} loss':round(total_loss,4),
                           f'valid {train_args["metric_fn"]}': round(val_metric,4),
                           'lr': round(math.log10(scheduler.lr[0]),4),
                           })
                
            # evaluate on testset
            model = Model(model_args).to(device)
            state_dict = torch.load(os.path.join(save_path,f'./best_fold{fold}.pt'))
            model.load_state_dict(state_dict)
            model.eval()
            test_metric = evaluate(testloader,model,device,metric_fn,metric_dtype,data_args['task'])
            results.append(test_metric)

            print(f'best epoch {best_epoch} for fold {fold}, val {train_args["metric_fn"]}:{best}, test: {test_metric}')
            wandb.finish()
    return results


if __name__=='__main__':

    import sys
    config_path = sys.argv[1]
    config = json.load(open(config_path,'r'))
    data_args = config['data']
    train_args = config['train']
    train_args['data_name'] = config_path.split('/')[-1].strip('.json')
    model_args = config['model']
    seed = config['seed']
    if not isinstance(seed,list):
        seed = [seed]
    
    print(config)
    results = train(data_args,train_args,model_args,seed)
    print(f'average performance: {np.mean(results)}+/-{np.std(results)}')