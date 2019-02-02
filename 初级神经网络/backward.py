import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
MNIST_data_folder="D:\\input_date.py"  
mnist=input_data.read_data_sets(MNIST_data_folder,one_hot=True)
import forward
import os
input_node=784
output_node=10
learning_rate_base=0.05
learning_rate_decay=0.96
steps=20000
MODEL_SAVE_PATH="./model/"
MODEL_NAME="mnist_model"
def backward(mnist):
    x=tf.placeholder(tf.float32,[None,input_node])
    y_=tf.placeholder(tf.float32,[None,output_node])
    y=forward.forward(x)
    global_step=tf.Variable(0)
    learning_rate= tf.train.exponential_decay(learning_rate_base,
        global_step,
        100,
        learning_rate_decay,
        staircase=True)
    los= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(los)
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    saver=tf.train.Saver()
    if_correct=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuarcy=tf.reduce_mean(tf.cast(if_correct,tf.float32))
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        for i in range(steps):
            xs, ys = mnist.train.next_batch(200)
            _, loss_value, accuracy1, step = sess.run([train_step, loss, accuarcy, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g.accuarcy is %g." % (step, loss_value,accuracy1))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    mnist=input_data.read_data_sets("D:\\input_date.py", one_hot=True)
    backward(mnist)

main()


        


