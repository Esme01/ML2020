import utils
import numpy as np
from matplotlib import pyplot as plt
x_train_path = 'data/X_train'
y_train_path = 'data/Y_train'
x_test_path = 'data/X_test'
output_path = 'data/output_{}.csv'
X_train,Y_train,X_test = utils.read_data(x_train_path,y_train_path,x_test_path)
#print(X_train.shape)
#print(X_test.shape)

X_train,X_mean,X_std = utils.normalize_data(X_train,is_train=True)
X_test,_,_ = utils.normalize_data(X_test,is_train=False,cols = None,X_mean=X_mean,X_std = X_std)
ratio = 0.1
X_train,Y_train,X_valid,Y_valid = utils.split_train_and_valid(X_train,Y_train,1-ratio)

train_size = X_train.shape[0]
#print(train_size)
valid_size = X_valid.shape[0]
#print(valid_size)
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
####training####
w = np.zeros((X_train.shape[1],))
b = np.zeros((1,))

epoches = 20
batch_size = 8
lr = 0.2

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
step  = 1
for epoch in range(epoches):
    X_train,Y_train = utils.shuffle(X_train,Y_train)

    for i in range(int(np.floor((train_size/batch_size)))):
        X = X_train[i*batch_size:(i+1)*batch_size]
        Y = Y_train[i*batch_size:(i+1)*batch_size]

        #w,b = utils.gradient_descent(X,Y,w,b,lr)
        w_grad,b_grad = utils.gradient_descent(X,Y,w,b)
        w -= lr/np.sqrt(step) * w_grad
        b -= lr / np.sqrt(step) * b_grad
        step += 1
    y_train_pred = utils.f(X_train,w,b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(utils.accruacy(Y_train_pred,Y_train))
    train_loss.append(utils.cross_entropy_loss(y_train_pred,Y_train)/train_size)

    y_valid_pred = utils.f(X_valid, w, b)
    Y_valid_pred = np.round(y_valid_pred)
    valid_acc.append(utils.accruacy(Y_valid_pred, Y_valid))
    valid_loss.append(utils.cross_entropy_loss(y_valid_pred, Y_valid) / valid_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Validation loss: {}'.format(valid_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Validation accuracy: {}'.format(valid_acc[-1]))


# Loss curve
plt.plot(train_loss)
plt.plot(valid_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(valid_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()


predictions = utils.predict(X_test, w, b)
with open(output_path.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(x_test_path) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])

