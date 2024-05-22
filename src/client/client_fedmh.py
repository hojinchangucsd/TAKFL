import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast

mixed_precision_training = False
class Client_FedMH(object):
    def __init__(self, name, model, local_bs, local_ep, optim, lr, momentum, local_wd, scheduler, device, 
                train_dl_local = None, test_dl_local = None, nlp=False):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.optim = optim
        self.lr = lr
        self.momentum = momentum 
        self.local_wd = local_wd
        self.scheduler = scheduler
        self.device = device 
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0 
        self.count = 0 
        self.save_best = True 
        self.nlp = nlp
        
    def train(self, is_print = False):
        #self.net = torch.nn.DataParallel(self.net)
        self.net.to(self.device)
        self.net.train()
        
        #optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        if self.optim == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.local_wd)
        elif self.optim == 'adamw':
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.local_wd)
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        
        if self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.local_ep, eta_min=self.lr/100)
        elif self.scheduler == "cosineW":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.local_ep//2, T_mult=1, eta_min=self.lr/100, last_epoch=-1)
        
        if mixed_precision_training:
            scaler = GradScaler()
        
        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, data in enumerate(self.ldr_train):
                if data is None:
                    continue

                if self.nlp: 
                    data_in = {k: v.to(self.device, non_blocking=True) for k, v in data.items()}

                else: 
                    data_in, labels = data
                    data_in, labels = data_in.to(self.device), labels.to(self.device)
                    labels = labels.type(torch.LongTensor).to(self.device)
                
                self.net.zero_grad()
                optimizer.zero_grad()
                
                if mixed_precision_training:
                    with autocast():
                        if not self.nlp: 
                            log_probs = self.net(data_in)
                            loss = self.loss_func(log_probs, labels)
                        else: 
                            outputs = self.net(**data_in)
                            loss = outputs.loss
                
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if not self.nlp: 
                        log_probs = self.net(data_in)
                        loss = self.loss_func(log_probs, labels)
                    else: 
                        outputs = self.net(**data_in)
                        loss = outputs.loss
                    loss.backward() 
                    optimizer.step()
                
                batch_loss.append(loss.item())
            
            if self.scheduler != "none":
                scheduler.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        self.net.to('cpu')
        return sum(epoch_loss) / len(epoch_loss)
    
    def get_state_dict(self):
        return self.net.state_dict()
    def get_best_acc(self):
        return self.acc_best
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
    
    def inference(self, public_ds):
        public_dl = torch.utils.data.DataLoader(public_ds, batch_size=128, shuffle=False, drop_last=False)
        self.net.eval()
        self.net.to(self.device)
        
        outs = []
        with torch.no_grad(): 
            for data, _,_ in public_dl:
                if self.nlp: 
                    data = {k:v.to(self.device, non_blocking=True) for k,v in data.items()}
                    out = self.net(**data)
                    outs.append(out.logits)
                else: 
                    data = data.to(self.device)
                    out = self.net(data)
                    outs.append(out.detach().cpu())

        self.net.cpu()
        outputs = torch.cat(outs)
            #print(f'out {out.shape}, output: {outputs.shape}')
        return outputs
    
    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy
    
    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy
    
    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy