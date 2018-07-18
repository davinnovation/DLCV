import torch
from torch import nn
from torch import optim
import json
import os
from misc import get_args
from loader import get_loader
from models import *

class Trainer(object):
    def __init__(self, model, train_loader, eval_loader , config):
        self.config = config
        self.optimizer = self.get_optimizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.history = self.get_history(config)
    
    def weights_init(m):
        if isinstance(m,nn.Conv2d):
            xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)
        elif isinstance(m,nn.Linear):
            xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)
    
    def init_model(self):
        self.model.apply()

    def get_optimizer(self):
        if self.config.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(),
                                        lr=self.config.lr,
                                        betas=(self.config.beta1,self.config.beta2),
                                        weight_decay = self.config.weight_decay,
                                        )
        elif self.config.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(),
                                      lr=self.config.lr,
                                      momentum=self.config.momentum,
                                      weight_decay = self.config.weight_decay
                                        )
        else:
            raise ValueError
        return optimizer

    def get_history(self,config):
        history_num = len(os.listdir(self.config.log_path))
        history = {}
        history['name']= "history.%02d.json"%history_num
        history['max_val_acc'] = 0
        history['history'] = {}
        return history

    def log_history(self,epoch, step, loss, acc, train=True):
        msg = "epoch: %02d | step: %05d | loss: %8f | acc: %5f | type: %s"%(epoch,step,loss,acc,"training" if train else "evaluating")
        print(msg)
        if train:
            if not epoch in self.history['history']:
                self.history['history'][epoch] = {'train':{},'val':{}}
            self.history['history'][epoch]['train'][step] = {'acc':acc,'loss':loss}
        else:
            self.history['history'][epoch]['val'][step] = {'acc':acc, 'loss':loss}
        json.dump(self.history,open(os.path.join(self.config.log_path, self.history['name']),"w"))

    def evaluate(self,epoch,step):
        self.model.eval()
        if not os.path.exists(self.config.save_root):
            os.mkdir(self.config.save_root)
        acc = 0
        n_items = 0
        mean_loss = 0
        n_step = 0
        for data, label in self.eval_loader:
            data, label = data.to(self.device), label.to(self.device)
            logit = self.model(data)
            loss = self.criterion(logit, label)
            acc += (torch.argmax(logit, dim=1) == label).sum().item()
            n_items += data.size(0)
            mean_loss += loss.item()*data.size(0)
        mean_loss/=n_items
        acc /= n_items
        self.log_history(epoch,step,mean_loss,acc,train=False)
        if acc > self.history['max_val_acc']:
            self.history['max_val_acc']=acc
            self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(),os.path.join(self.config.save_root,"%s.pth"%self.config.model_name))

    def train(self):
        n_epoch = self.config.n_epoch
        max_acc = 0
        for epoch in range(n_epoch):
            epoch +=1
            n_items, acc, mean_loss =0, 0, 0
            for step, (data, label) in enumerate(self.train_loader):
                step+=1

                # set optimizer to zero-gradient state & set model to training state
                self.optimizer.zero_grad()
                self.model.train()

                # tensor with cuda or cpu
                data, label = data.to(self.device), label.to(self.device)

                # feed forward
                logit = self.model(data)

                # get loss and backpropagate
                loss = self.criterion(logit,label)
                loss.backward()

                # update parameters
                self.optimizer.step()

                # calculate accuracy and mean loss
                acc += (torch.argmax(logit,dim=1) == label).sum().item()
                n_items += data.size(0)
                mean_loss += loss.item()*data.size(0)

                # checkpoint
                if step % self.config.check_step == 0:
                    acc /= n_items
                    mean_loss /= n_items
                    self.log_history(epoch,step,mean_loss,acc,train=True)
                    n_items, acc, mean_loss = 0, 0, 0

            # evaluate
            self.evaluate(epoch, step)


if __name__ == '__main__':
    config = get_args()
    train_loader = get_loader(config,train=True)
    test_loader = get_loader(config,train=False)
    if config.model_name.lower() == 'alexnetoriginal':
        model = AlexNetOriginal()
    elif config.model_name.lower() == 'alexnetbn':
        model = AlexNetBn()
    if config.pretrained_path is not None:
        model.load_state_dict(torch.load(config.pretrained_path))
    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()
