import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
data_dir="D:\python2\cifar-10\cifar-10-batches-py"
save_path="./modelcifar10/"
model_name="cifar-10_model"
batch_size=100
learning_rate_base=0.001
learning_rate_decay=0.96
max_size=49998
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict1=unpickle('D:\python2\cifar-10\cifar-10-batches-py\data_batch_1')
dict2=unpickle('D:\python2\cifar-10\cifar-10-batches-py\data_batch_2')
dict3=unpickle('D:\python2\cifar-10\cifar-10-batches-py\data_batch_3')
dict4=unpickle('D:\python2\cifar-10\cifar-10-batches-py\data_batch_4')
dict5=unpickle('D:\python2\cifar-10\cifar-10-batches-py\data_batch_5')

images1=dict1[b'data']
labels1=dict1[b'labels']

images2=dict2[b'data']
labels2=dict2[b'labels']

images3=dict3[b'data']
labels3=dict3[b'labels']

images4=dict4[b'data']
labels4=dict4[b'labels']

images5=dict5[b'data']
labels5=dict5[b'labels']

images_=np.concatenate((images1,images2,images3,images4,images5),axis=0)   #读取数据
labels_=np.concatenate((labels1,labels2,labels3,labels4,labels5),axis=0)   #读取数据

def knn_distance(a,b):    #进行每张图片的距离计算，返回一个距离
    distance=np.sum(abs(a-b))
    return distance

def get_label(n):
    arr1=[0,1,2,3,4]
    arr1.sort(key=lambda x:knn_distance(images_[n],images_[x]))  #对数组进行降序排列
    for i in list(range(5,50000)):
        if i == n:
            continue
        elif knn_distance(images_[i],images_[n])<knn_distance(images_[arr1[4]],images_[n]):
            arr1[4]=i
            arr1.sort(key=lambda x:knn_distance(images_[n],images_[x]))     #递归寻找五个距离最小点
    arr2=np.zeros(10)
    for i in arr1:
        arr2[labels_[i]]=arr2[labels_[i]]+1
    return np.argmax(arr2)

for i in range(500):
    start=i*batch_size
    end=(i+1)*batch_size-1
    sum=0
    for j in range(start,end):
        if get_label(j)==labels_[j]:
            sum=sum+1
    print("the accuracy in this batch size is %g."%(sum/batch_size))
