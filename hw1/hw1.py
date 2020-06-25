import numpy as np
import pandas as pd
from get_data import DataLoader
from LR import LR
if __name__ == "__main__":
    file_name = 'data/train.csv'
    test_file = 'data/test.csv'
    data_loader = DataLoader(file_name)
    raw_data = data_loader.get_data_as_df()
    data_dict = data_loader.get_data_by_month(raw_data)
    train_data,labels,mean_x,std_x = data_loader.get_final_data(data_dict)
    #x_train_set,label_train_set,\
    #x_validation,label_validation = data_loader.split_train_and_valid(train_data,labels)
    test_data = pd.read_csv(test_file, header=None, encoding='big5')
    test_data = test_data.iloc[:, 2:]
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()
    test_x = np.empty([240,18*9],dtype = float)
    for i in range(240):
        test_x[i] = test_data[18*i:18*(i+1),:].reshape(1,-1)

    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

    linear_model = LR(train_data,labels)
    linear_model.train()
    linear_model.get_predict_csv(test_x)
