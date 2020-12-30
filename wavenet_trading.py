# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:35:23 2020

@author: allen
"""
import time, copy
from collections import OrderedDict
from functools import reduce
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import new_dataloader
import torch.utils.data as Data
import matplotlib.pyplot as plt
import test 
import test_profit
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
import pandas as pd
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams['pdf.fonttype'] = 42
# Hyper Parameters
EPOCH = 100       # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 128
INPUT_SIZE = 150         # rnn input size / image width
LR = 0.01               # learning rate
torch.cuda.empty_cache()
#train_data, train_label, test_data, test_label = dataloader.read_data()
def DataTransfer(train_data, train_label, test_data, test_label):
    
    train_data = torch.FloatTensor(train_data).cuda()
    train_label=torch.LongTensor(train_label).cuda() #data型態轉換@
    test_data=torch.FloatTensor(test_data).cuda()
    test_label=torch.LongTensor(test_label).cuda()
    torch_dataset_train = Data.TensorDataset(train_data, train_label)
    loader_train = Data.DataLoader(
            dataset=torch_dataset_train,      # torch TensorDataset format
            batch_size = BATCH_SIZE,      # mini batch size
            shuffle = True,               
            )

    torch_dataset_test = Data.TensorDataset(test_data, test_label)
    loader_test = Data.DataLoader(
            dataset=torch_dataset_test,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle = False,              
            )
    return loader_train, loader_test

                     # the target label is not one-hotted

class Wave_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        self.convs.append(nn.BatchNorm1d(out_channels))
        #self.convs.append(nn.MaxPool1d(2))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate)
                )
            self.filter_convs.append(nn.BatchNorm1d(out_channels))      
            #self.filter_convs.append(nn.MaxPool1d(2))
            self.gate_convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(nn.BatchNorm1d(out_channels))              
            #self.gate_convs.append(nn.MaxPool1d(2))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))
            self.convs.append(nn.BatchNorm1d(out_channels))
            #self.convs.append(nn.MaxPool1d(2))

            

    def forward(self, x):
        x = self.convs[0](x)
        #print("res",x.shape)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res

class Classifier(nn.Module):
    def __init__(self, inch=1, kernel_size=3):
        super().__init__()
        #self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.wave_block1 = Wave_Block(inch, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.fc = nn.Linear(65536 , 768)
        self.fc2 = nn.Linear(768,6)

    def forward(self, x):
        #x = x.permute(0, 2, 1)
        #print(x.shape)
        x = self.wave_block1(x)
        #print(x.shape)
        x = self.wave_block2(x)
        #print(x.shape)
        x = self.wave_block3(x)
        #print(x.shape)

        x = self.wave_block4(x)
        #x = x.permute(0, 2, 1)
        #x, _ = self.LSTM(x)
        #print(x.shape)
        x = x.view(x.size(0),-1)
        
        x = self.fc(x)
        x = self.fc2(x)
        #print(x.shape)
        return x

# training and testing
def model_train(loader_train,loader_test):
    
    """
    CNN1_class = CNN_classsification1().cuda()
    optimizer = torch.optim.Adam(CNN1_class.parameters(), lr=LR)   # optimize all cnn parameters
    loss_xe = nn.CrossEntropyLoss().cuda()          
    """
    msresnet = Classifier() #channel 設定
    msresnet = msresnet.cuda()
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    optimizer = torch.optim.Adam(msresnet.parameters(), lr = LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80, 95], gamma=0.9)
    
    train_loss =[]
    train_acc=[]
    winrate = []
    big_profit  = -100000

    for epoch in range(EPOCH):
            total_train = 0
            correct_train = 0
            action_list=[]
            action_choose=[]
            msresnet.train()
            scheduler.step()
            
            for step, (batch_x, batch_y) in enumerate(loader_train):   # 分配 batch data, normalize x when iterate train_loader
                output = msresnet(batch_x)             # cnn output
                loss = criterion(output, batch_y)   # mseloss
                
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients
                _, predicted = torch.max(output, 1)
                #print("預測分類 :",predicted)
                total_train += batch_y.nelement()
                correct_train += predicted.eq(batch_y).sum().item()
            train_accuracy = 100 * correct_train / total_train 
            train_loss.append(loss.item())                                          
            print('Epoch {}, train Loss: {:.5f}'.format(epoch+1, loss.item()), "Training Accuracy: %.2f %%" % (train_accuracy))
            
            train_acc.append(train_accuracy)

            for step, (batch_x, batch_y) in enumerate(loader_test):
                output = msresnet(batch_x).cuda()
                loss = criterion(output, batch_y).cuda()
                _, predicted = torch.max(output, 1)
                action_choose = predicted.cpu().numpy()
                action_choose = action_choose.tolist()
                action_list.append(action_choose)
            action_list =sum(action_list, [])
            print("幾個 action :",len(action_list))
            
            profit = test_profit.reward(action_list) #讓model跑更快
            winrate.append(profit)
            #test_accuracy = 100 * correct_test / total_test          #avg_accuracy = train_accuracy / len(train_loader)
            print('Epoch {}, test Loss: {:.5f}'.format(epoch+1, loss.item()), "Testing winrate: %.2f %%" % (profit))
            if train_accuracy >= 0 and profit > big_profit :
                big_profit = profit
                torch.save(msresnet,"2015_2016wavenet.pkl")
            
    draw(train_loss,train_acc,winrate)
def draw(train_loss, train_acc,winrate):
    #plt.title('CNN_Net Classificaiton for pair_trading')
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Train_loss')
    plt.plot(train_loss)
    plt.tight_layout()
    plt.savefig('Pair Trading WaveNet loss(2016).png')
    plt.show()        
    plt.close()

    #plt.title("CNN_Net Classificaiton for pair_trading")
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Train_Accuaracy(%)')
    plt.plot(train_acc)
    plt.tight_layout()
    plt.savefig('Pair Trading WaveNet auc(2016).png')
    plt.show()        
    plt.close()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Winrate(%)')
    plt.plot(winrate)
    plt.tight_layout()
    plt.savefig('Pair Trading WaveNet Validation Winrate(2016).png')
    plt.show()        
    plt.close()
    
    df = pd.DataFrame({'loss': train_loss,
                   'acc': train_acc,
                   'winrate': winrate})
    df.to_csv("2015_2016wavenet.csv",index=False)

if __name__=='__main__':
    choose = 1
    if choose == 0 :
        train_data, train_label, test_data, test_label = new_dataloader.read_data()
        loader_train,loader_test = DataTransfer(train_data, train_label, test_data, test_label)
        model_train(loader_train,loader_test)
    else :
    #model_train()
        test.test_reward()