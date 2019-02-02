import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
MNIST_data_folder="D:\\input_date.py"  
mnist=input_data.read_data_sets(MNIST_data_folder,one_hot=True)
def get_weight(shape):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    return w
def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b
def conv2d(input,w):
    return tf.nn.conv2d(input,w,strides=[1,1,1,1],padding='SAME')
def max_pooling_2x2(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])
w1=get_weight([5,5,1,32])
b1=get_bias([32])
h1_conv=tf.nn.relu(conv2d(x_image,w1)+b1)
h1=max_pooling_2x2(h1_conv)

w2=get_weight([5,5,32,64])
b2=get_bias(64)
h2_conv=tf.nn.relu(conv2d(h1,w2)+b2)
h2=max_pooling_2x2(h2_conv)

w_out=get_weight([7*7*64,1024])
b_out=get_bias([1024])
h2_out=tf.reshape(h2,[-1,7*7*64])

h2_outs=tf.nn.relu(tf.matmul(h2_out,w_out)+b_out)

w3=get_weight([1024,10])
b3=get_bias([10])
y=tf.nn.softmax(tf.matmul(h2_outs,w3)+b3)

loss=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
correct_predicton=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_predicton,tf.float32))
with tf.Session() as sess:
    init=tf.initialize_all_variables()
    sess.run(init)
    for i in range(10000):
        batch=mnist.train.next_batch(200)
        _,loss_,accuracy_=sess.run([train_step,loss,accuracy] ,feed_dict={x:batch[0] , y_:batch[1]})
        if i % 100 == 0:
            print("after %d steps of training,the loss is %g. ,accuracy is %g."%(i,loss_,accuracy_))