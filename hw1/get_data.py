import numpy as np
import pandas as pd
import math

class DataLoader:
    def __init__(self,file):
        self.file_name = file

    def get_data_as_df(self):
        data = pd.read_csv(self.file_name,encoding='big5')
        data = data.iloc[:,3:]  #去掉前三列
        data[data=='NR'] = 0   #RAINFALL转化成数值
        return data.to_numpy()   #data 4320行 = 12 * 18 * 20

    def get_data_by_month(self,data):
        '''
        :param data: numpy矩阵
        每个月共20*24 = 480小时，将每个月的数据转化成18*480的矩阵
        :return:处理之后的dict
        '''
        data_dict = {}
        day_start = 0
        for month in range(12):
            month_data = data[day_start:day_start+18,:]
            day_start += 18
            for day in range(1,20):
                one_day_data = data[day_start:day_start+18,:]
                day_start += 18
                month_data = np.concatenate((month_data,one_day_data),axis=1)
            data_dict[month] = month_data
        return data_dict

    def get_final_data(self,data):
        '''
        将data进行分组，每一组数据是前九个小时18*9的feature,第10个小时的PM2.5为label
        对于18*480的月数据，可以分成471组数据，每组数据shape为18*10
        总数据量为471*12组数据
        :param data: dict get by function get_data_by_month
        :return: processed data X.shape = [12*471,18*9] label.shape = [12*471,1]
        '''
        X = np.empty([12*471,18*9],dtype=float)
        label = np.empty([12*471,1],dtype=float)
        for month in range(12):
            for i in range(471):
                X[month*471 + i] = data[month][:,i:i+9].reshape(1,-1)
                label[month*471 + i,0] = data[month][9,i+9]


        mean_x = np.mean(X, axis=0)  # 18 * 9
        std_x = np.std(X, axis=0)  # 18 * 9
        for i in range(len(X)):  # 12 * 471
            for j in range(len(X[0])):  # 18 * 9
                if std_x[j] != 0:
                    X[i][j] = (X[i][j] - mean_x[j]) / std_x[j]

        return X,label,mean_x,std_x

    def split_train_and_valid(self,X,y):
        x_train_set = X[: math.floor(len(X) * 0.8), :]
        y_train_set = y[: math.floor(len(y) * 0.8), :]
        x_validation = X[math.floor(len(X) * 0.8):, :]
        y_validation = y[math.floor(len(y) * 0.8):, :]
        return x_train_set,y_train_set,x_validation,y_validation


