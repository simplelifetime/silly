import tensorflow as tf
import forward,backward
import os
import numpy as np
data_dir="D:\python2\cifar-10\cifar-10-batches-py"
save_path="./modelcifar10/"
model_name="cifar-10_model"
dict=backward.unpickle('D:\python2\cifar-10\cifar-10-batches-py\\data_batch_1')
images=dict[b'data']
labels=dict[b'labels']
images_in=np.reshape(images,[-1,32,32,3])
def test():
    # images_ = tf.placeholder(tf.float32, [None, 32, 32, 3])
    # labels_ = tf.placeholder(tf.float32, [None])
    # result=forward.forward(images_,0.0)
    # label=tf.one_hot(tf.cast(labels_,tf.int64),10)
    # correct_prediction=tf.equal(tf.argmax(label,1),tf.argmax(result,1))
    # accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))                                   
    images_=tf.placeholder(tf.float32,[None,32,32,3])
    labels_=tf.placeholder(tf.float32,[None])
    labels_in=tf.one_hot(tf.cast(labels_,tf.int64),10)
    labels_out=forward.forward(images_,0.0)
    if_correct=tf.equal(tf.argmax(labels_out,1),tf.argmax(labels_in,1))
    accuracy=tf.reduce_mean(tf.cast(if_correct,tf.float32))
    saver=tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(save_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        acc=sess.run(accuracy,feed_dict={images_:images_in,labels_:labels})
        print(acc)

test()

    