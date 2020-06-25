import numpy as np
import pandas as pd
import csv


class LR:
    def __init__(self,train_data,labels,lr=100,iter=2000,lamd = 0):
        self.train_data = train_data
        self.label = labels
        self.lr = lr
        self.iter = iter
        self.eps = 0.0000000001
        self.lamd = lamd
    def train(self):
        w = np.zeros([18*9 + 1,1]) #w和b合并
        adagrad = np.zeros([18*9+1,1])
        X = np.concatenate((np.ones([12 * 471, 1]), self.train_data), axis=1).astype(float)
        for i in range(self.iter):
            loss = np.sqrt(np.sum(self.label-np.dot(X,w))**2)/(471*12)
            if(i%100 == 0): print(str(i) + ":" + str(loss))

            grad = 2 * np.dot(X.T,np.dot(X,w) - self.label)
            adagrad += grad**2
            w -= self.lr * grad / np.sqrt(adagrad+self.eps)

        np.save('weight.npy',w)

    def predict(self,test_data):
        w = np.load('weight.npy')
        return np.dot(test_data,w)

    def get_predict_csv(self,test_data):
        with open('submit.csv', mode='w', newline='') as submit_file:
            csv_writer = csv.writer(submit_file)
            header = ['id', 'value']
            print(header)
            csv_writer.writerow(header)
            ans_y = self.predict(test_data)
            for i in range(240):
                row = ['id_' + str(i), ans_y[i][0]]
                csv_writer.writerow(row)
                print(row)


