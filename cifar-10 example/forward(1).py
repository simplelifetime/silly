import tensorflow as tf
def get_weight1(shape,lamb):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    tf.add_to_collection('w_loss',tf.contrib.layers.l2_regularizer(lamb)(w))
    return w

def get_weight2(shape):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    # tf.add_to_collection('w_loss',tf.contrib.layers.l2_regularizer(lamb)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b

def conv2d(input,w):
    return tf.nn.conv2d(input,w,strides=[1,1,1,1],padding='SAME')

def max_pooling_2x2(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(input,lamb):
    w1=get_weight2([5,5,3,32])
    b1=get_bias([32])
    h1_conv=tf.nn.relu(conv2d(input,w1)+b1)
    h1=max_pooling_2x2(h1_conv)

    w2=get_weight2([5,5,32,64])
    b2=get_bias(64)
    h2_conv=tf.nn.relu(conv2d(h1,w2)+b2)
    h2=max_pooling_2x2(h2_conv)

    w_out=get_weight2([8*8*64,1024])
    b_out=get_bias([1024])
    h2_out=tf.reshape(h2,[-1,8*8*64])

    h2_outs=tf.nn.relu(tf.matmul(h2_out,w_out)+b_out)
    # h2_oute=tf.nn.dropout(h2_outs,keep_prob)

    w3=get_weight1([1024,10],lamb)
    b3=get_bias([10])
    y_final=tf.nn.softmax(tf.matmul(h2_outs,w3)+b3)
    return y_final
