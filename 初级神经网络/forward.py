import tensorflow as tf
def get_weight(shape):
    w=tf.Variable(tf.truncated_normal(shape))
    return w

def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b

def forward(x):
    w1=get_weight([784,300])
    w2=get_weight([300,10])
    b1=get_bias([300])
    b2=get_bias([10])
    t1=tf.nn.relu(tf.matmul(x,w1)+b1)
    y=tf.matmul(t1,w2)+b2
    return y