import tensorflow as tf
import forward,cifar10_input,cifar10
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

images_=np.concatenate((images1,images2,images3,images4,images5),axis=0)
labels_=np.concatenate((labels1,labels2,labels3,labels4,labels5),axis=0)
images_in=np.reshape(images_,[-1,32,32,3])
def backward():
    images = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels = tf.placeholder(tf.float32, [None])
    lamb=tf.placeholder(tf.float32)
    result=forward.forward(images,lamb)
    label=tf.one_hot(tf.cast(labels,tf.int64),10)
    cross_entropy =-tf.reduce_sum(label*tf.log(tf.clip_by_value(result,0,1000)),name='cross')
    tf.add_to_collection('w_loss',cross_entropy) 
    loss=tf.add_n(tf.get_collection('w_loss'))
    global_step=tf.Variable(0)
    learning_rate= tf.train.exponential_decay(learning_rate_base,
        global_step,
        100,
        learning_rate_decay,
        staircase=True)
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    correct_prediction=tf.equal(tf.argmax(label,1),tf.argmax(result,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))                                   
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)

        ckpt=tf.train.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)   
        for i in range(5000):
            start=(i*batch_size+1)%max_size
            end=min(start+batch_size,max_size)
            batch_images=images_in[start:end]
            batch_labels=labels_[start:end]
            _,loss_,accuracy_,steps=sess.run([train_step,loss,accuracy,global_step],feed_dict={images:batch_images, labels:batch_labels,lamb:0.0001})
            # if i % 50 == 0:
            print("after %d of training,loss is %g.,accuracy is %g."%(steps,loss_,accuracy_))
            # if i % 500 ==0: 
            #     saver.save(sess,os.path.join(save_path, model_name), global_step=global_step)

if __name__ == "__main__":
    # backward()
    print(images_[1])
    s1=np.reshape(images_[1],[1,32,32,3])
    print(s1)
    plt.imshow(s1)
    plt.show()