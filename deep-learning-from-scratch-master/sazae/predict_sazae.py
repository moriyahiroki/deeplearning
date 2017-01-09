import sys, os
sys.path.append(os.pardir)
import numpy as np
from neural_net import TwoLayerNet

# トレーニングデータとテストデータを用意

# サザエさんのじゃんけんデータ読み取り用
data = []

# 過去の参照データ数
sample_num = 10
hidden_num = 20
training_num = 1000

x_data = np.array([])
t_data = np.array([])

for line in open('sazae_history_exp_3.txt','r'):
    line = line.replace('\n','')
    line = line.replace('\r','')
    data.append(int(line))

for i in range(sample_num,1270):
    x_data = np.append(x_data,np.array(data[i-sample_num:i]))
    t_tmp = np.array([0,0,0])
    t_tmp[data[i]] = 1
    t_data = np.append(t_data,np.array(t_tmp))

x_data = x_data.reshape(-1,sample_num)
t_data = t_data.reshape(-1,3)

'''
for line in open('x_data.txt','r'):
    line = line.replace('\n','')
    line = line.replace('\r','')
    data.append(int(line))

for line in open('t_data.txt', 'r'):
    line = line.replace('\n', '')
    line = line.replace('\r', '')
    t_tmp = np.array([0,0,0])
    t_tmp[int(line)] = 1
    t_data = np.append(t_data, np.array(t_tmp))

for i in range(sample_num, len(data)):
    x_data = np.append(x_data, np.array(data[i-sample_num:i]))

x_data = x_data.reshape(-1,sample_num)
t_data = t_data.reshape(-1,3)
t_data = t_data[sample_num:1271]

print(x_data.shape)
print(t_data.shape)
'''
# トレーニングデータ
x_train = x_data[:training_num]
t_train = t_data[:training_num]

# テストデータ
x_test = x_data[training_num:1270]
t_test = t_data[training_num:1270]

network = TwoLayerNet(input_size=sample_num, hidden_size=hidden_num, output_size=3)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.05

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

