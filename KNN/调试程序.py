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

def findlabel(n):
    arr1=[0,1,2,3,4,5,6]
    arr1.sort(key=lambda x:knn_distance(images_[n],images_[x]),reverse=True)  #对数组进行降序排列 
    for i in list(range(7,50000)):
        if i == n:
            continue
        else:
            if knn_distance(images_[i],images_[1])>=knn_distance(images_[arr1[6]],images_[n]):
                continue
            else:
                for j in [5,4,3,2,1,0]:
                    if knn_distance(images_[i],images_[n])<=knn_distance(images_[arr1[j]],images_[n]):
                        if j !=0:
                            arr1[j+1]=arr1[j]
                        else:
                            arr1[j+1]=arr1[j]
                            arr1[0]=i
                    else:
                        arr1[j+1]=i
                        break
    # print(arr1)
    arr2=np.zeros(10)
    for i in arr1:
        arr2[labels_[i]]=arr2[labels_[i]]+1
    # print(arr2)
    # print(np.argmax(arr2),labels_[n])
    return np.argmax(arr2)

def main():
    for i in range(200):
        start=i*50+7
        end=min(49999,start+50)
        sum=0
        for j in range(start,end):
            if findlabel(j)==labels_[j]:
                sum=sum+1
        print("in batch %d,the result is %g"%(i,sum/50))

# main()
plt.rcParams['figure.figsize']=(10.0,8.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'
for i in range(10):
    s=np.reshape(images_[i],[32,32,3])
    plt.imshow(s.astype('uint8'))
    plt.axis('off')
    plt.show()

