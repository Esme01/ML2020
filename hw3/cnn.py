import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time

def getData(path,re_label):
    imgs = sorted(os.listdir(path))
    #512 * 512 * 3
    X = np.zeros((len(imgs),128,128,3),dtype=np.uint8)
    y = np.zeros((len(imgs)),dtype=np.uint8)
    i = 0
    for img in imgs:
        tmp = Image.open(os.path.join(path+img))
        tmp = np.array(tmp.resize((128,128)))
        X[i,:,:] = tmp
        if re_label:
            #filename : label_id.jpg
            y[i] = int(img.split('_')[0])
        i+=1
    if re_label:
        return X,y
    else:
        return X

path = "D:\\FILES\\food-11\\"
print("Reading data")
train_x, train_y = getData(os.path.join(path, "training\\"), True)
print(train_x[0].shape) #(128,128,3)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = getData(os.path.join(path, "validation\\"), True)
print("Size of validation data = {}".format(len(val_x)))
test_x = getData(os.path.join(path, "testing\\"), False)
print("Size of Testing data = {}".format(len(test_x)))

# training set数据增广
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # 水平翻转
    transforms.RandomRotation(15), # 旋转
    transforms.ToTensor(), # 转为tensor并normalization
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
batch_size = 64
train_set = ImgDataset(train_x, train_y, train_transform)
print(train_set[0][0].size()) #(3,128,128)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            nn.Conv2d(3,64,3,1,1), # output:(64,128,128)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), # output:(64,64,64)

            nn.Conv2d(64,128,3,1,1), #output:[128,64,64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), #[128,32,32]

            nn.Conv2d(128,256,3,1,1), #[256,32,32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), #[256,16,16]

            nn.Conv2d(256,512,3,1,1), #[512,16,16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), #[512,8,8]

            nn.Conv2d(512,512,3,1,1), #[512,8,8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),#[512，4，4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)  #11个分类
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)  # flatten
        return self.fc(out)


model = Classifier().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
num_epoches = 30

for epoch in range(num_epoches):
    epoch_start = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i,data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred,data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    # validation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.data.cpu().numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

            # 将结果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoches, time.time() - epoch_start, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

# 使用训练集和验证集重新训练
train_val_x = np.concatenate((train_x, val_x), axis=0) # train_x拼接val_x
train_val_y = np.concatenate((train_y, val_y), axis=0) # train_y拼接val_y
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因为是分类任务，所以使用交叉熵损失
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoches = 30

for epoch in range(num_epoches):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        #将结果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoches, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))


test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        # 预测值中概率最大的下标即为模型预测的食物标签
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)
#将预测结果写入 csv
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
