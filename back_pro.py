import numpy as np
import os
os.chdir('D:\\Work\\Pylearn')

dataset=np.load('boston_housing.npz')
train_data=dataset['x']#训练数据
y=dataset['y']#标记

#激活函数
def relu(x):
    return x if x>=0 else 0
#relu导函数
def relu_der(x):
    return 1 if x>0 else 0

#损失函数
def loss(output, label):
    return np.power(output-label, 2)

#使用反向传播算法来优化一个有两个隐藏层，每层各十个神经元的神经网络。网络的输入为一个十三维的向量，输出层只有一个节点
#隐藏层的每一个单元都使用relu作为激活函数

#前馈网络
def forward(input, weights):
    #每一个隐藏节点的线性输入
    linear_input={}
    linear_input['hidden1']=np.dot(weights['w1'], input).astype(np.float64)
    linear_input['hidden2']=np.dot(weights['w2'], linear_input['hidden1']).astype(np.float64)
    #每一个节点的输出
    outputs={}
    activation_vector=np.vectorize(relu)#将所定义的函数向量化，使其可以作用于array中的每一个元素上
    outputs['input']=input
    outputs['hidden1']=activation_vector(linear_input['hidden1']).astype(np.float64)#非线性化处理
    outputs['hidden2']=activation_vector(linear_input['hidden2']).astype(np.float64)#非线性化处理
    outputs['output']=np.dot(weights['w3'], outputs['hidden2']).astype(np.float64)
    return outputs, linear_input

#反馈网络
def backward(linear_input, outputs, weights,label):
    #求每一个节点激活函数的导数值
    linear_input_der={}
    for key, value in linear_input.items():
        linear_input_der[key]=np.vectorize(relu_der)(value)
    #损失函数对每一个节点线性输入的导数的反向传播
    fi={}
    fi['output']=2*(outputs['output']-label)*1#在此由于输出节点没有激活函数，因此乘1表示激活函数的导数为1
    fi['hidden2']=np.multiply(np.dot(fi['output'], weights['w3']), linear_input_der['hidden2'])
    fi['hidden1']=np.multiply(np.dot(fi['hidden2'], weights['w2']), linear_input_der['hidden1'])
    
    #计算各个参数的梯度
    w_der={}
    w_der['w3']=fi['output']*outputs['hidden2']
    w_der['w2']=np.dot(fi['hidden2'].reshape(1,-1).T, outputs['hidden1'].reshape(1,-1))
    w_der['w1']=np.dot(fi['hidden1'].reshape(1,-1).T, outputs['input'].reshape(1, -1))
    return w_der

#生成随机参数
weights={}
weights['w1']=np.random.uniform(low=-0.1, high=0.1, size=(10, 13))
weights['w2']=np.random.uniform(low=-0.1, high=0.1, size=(12, 10))
weights['w3']=np.random.uniform(low=-0.1, high=0.1, size=(1, 12))

#mini-batch梯度下降法优化参数
lr=1e-5#学习率
epochs=1000
batch_size=50#一共506个数据，一个batch50个数据
batch_num=506//50#每一轮的batch数量
loss_history=[]#记录每一epoch的平均损失

#stochastic 训练
sample_num=train_data.shape[0]
for epoch in range(epochs):
    loss_epoch=[]
    for (idx, sample) in enumerate(train_data):
        label=y[idx]
        (outputs, linear_input)=forward(sample, weights)#前馈网络
        weight_der=backward(linear_input, outputs, weights, label)#反馈网络
        loss_epoch.append(loss(outputs['output'], label))
        for key in weight_der.keys():
            weights[key]=weights[key]-lr*weight_der[key]/sample_num
    print(np.mean(loss_epoch))
    loss_history.append(np.mean(loss_epoch))
np.save('stochastic',loss_history)
temp=0

# #mini-batch进行训练
# sample_num=train_data.shape[0]
# for epoch in range(epochs):
#     loss_epoch=[]
#     #mini-batch
#     for batch_idx in range(batch_num):
#         der_tem={}
#         der_tem['w1']=np.zeros(shape=(10, 13))
#         der_tem['w2']=np.zeros(shape=(12, 10))
#         der_tem['w3']=np.zeros(shape=(1, 12))
#         #分别求每一个batch梯度以及损失
#         if batch_idx!=9:
#             for idx in range(batch_idx*batch_size, (batch_idx+1)*batch_size):
#                 sample=train_data[idx]
#                 label=y[idx]
#                 (outputs, linear_input)=forward(sample, weights)#前馈网络
#                 weight_der=backward(linear_input, outputs, weights, label)#反馈网络
#                 loss_epoch.append(loss(outputs['output'], label))
#                 #计算梯度，将其放入der_tem中
#                 for (key, value) in weight_der.items():
#                     der_tem[key]=der_tem[key]+weight_der[key]

#             #每一个batch更新一次权重
#             for key in der_tem.keys():
#                 weights[key]=weights[key]-lr*der_tem[key]/batch_size
#         else:#第10个batch内只有6个样本
#             for idx in range((batch_idx+1)*batch_size, 506):
#                 sample=train_data[idx]
#                 label=y[idx]
#                 (outputs, linear_input)=forward(sample, weights)#前馈网络
#                 weight_der=backward(linear_input, outputs, weights, label)#反馈网络
#                 loss_epoch.append(loss(outputs['output'], label))
#                 #计算梯度，将其放入der_tem中
#                 for (key, value) in weight_der.items():
#                     der_tem[key]=der_tem[key]+weight_der[key]

#             #每一个batch更新一次权重
#             for key in der_tem.keys():
#                 weights[key]=weights[key]-lr*der_tem[key]/6

#     loss_history.append(np.mean(loss_epoch))
#     print(np.mean(loss_epoch))
# np.save('mini_batch',loss_history)
# temp=0