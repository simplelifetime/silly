import tensorflow as tf
import numpy as np
layer_1=64
layer_2=128
layer_3=256
layer_4=64	
def get_weight_l2(shape,lamb):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.01))
    tf.add_to_collection('w_loss',tf.contrib.layers.l2_regularizer(lamb)(w))
    return w

def get_weight(shape):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.01))
    return w

def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b

def conv2d(input,w):
    return tf.nn.conv2d(input,w,strides=[1,1,1,1],padding='SAME')

def max_pool(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(inside,keep_prob,lamb):
    input=tf.reshape(inside,[-1,28,28,1])
    w1=get_weight([3,3,1,layer_1])
    b1=get_bias([layer_1])
    h1_conv=tf.nn.relu(conv2d(input,w1)+b1)
    h1=max_pool(h1_conv)

    w2=get_weight([5,5,layer_1,layer_2])
    b2=get_bias([layer_2])
    h2_conv=tf.nn.relu(conv2d(h1,w2)+b2)
    h2=max_pool(h2_conv)

    w3=get_weight([3,3,layer_2,layer_3])
    b3=get_bias([layer_3])
    h3_conv=tf.nn.relu(conv2d(h2,w3)+b3)

    w4=get_weight_l2([7*7*layer_3,layer_4],lamb)
    b4=get_bias([layer_4])
    h3_out=tf.reshape(h3_conv,[-1,7*7*layer_3])
    h4=tf.nn.relu(tf.matmul(h3_out,w4)+b4)
    h4_oute=tf.nn.dropout(h4,keep_prob)

    w5=get_weight_l2([layer_4,10],lamb)
    b5=get_bias([10])
    result=tf.nn.softmax(tf.matmul(h4,w5)+b5)
    return result
