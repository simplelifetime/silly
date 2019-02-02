import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import forward1
import os
MNIST_data_folder="D:\\input_date.py"  
mnist=input_data.read_data_sets(MNIST_data_folder,one_hot=True)
MODEL_SAVE_PATH="./modelCNN1/"
MODEL_NAME="mnist_model"
learning_rate_base=0.001
learning_rate_decay=0.98
def backward():
    x=tf.placeholder(tf.float32,[None,784])
    y_=tf.placeholder(tf.float32,[None,10])
    lamb=tf.placeholder(tf.float32)
    keep_prob=tf.placeholder(tf.float32)
    y=forward1.forward(x,keep_prob,lamb)
    losses=-tf.reduce_sum(y_*tf.log(y))
    tf.add_to_collection('w_loss',losses)
    loss=tf.add_n(tf.get_collection('w_loss'))
    global_step=tf.Variable(0)
    learning_rate= tf.train.exponential_decay(learning_rate_base,
        global_step,
        100,
        learning_rate_decay,
        staircase=True)
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    correct_predicton=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_predicton,tf.float32))
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        for i in range(10000):
            batch=mnist.train.next_batch(50)
            _,loss_,accuracy_,steps=sess.run([train_step,loss,accuracy,global_step] ,feed_dict={x:batch[0] , y_:batch[1],keep_prob:0.5,lamb:0.001})
            if i % 100 == 0:
                print("after %d steps of training,the loss is %g. ,accuracy is %g."%(steps,loss_,accuracy_))
            if i % 2000 ==0:
                saver.save(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
backward()
            