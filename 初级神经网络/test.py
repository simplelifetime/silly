import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import forward
model_path ="./model/"
model_name="mnist_model"
def test(mnist):
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32,[None,10])
    y = forward.forward(x)
    if_correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuarcy = tf.reduce_mean(tf.cast(if_correct,tf.float32))
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path) 
        # init = tf.initialize_all_variables()
                # sess.run(init)
        xs=mnist.test.images
        ys=mnist.test.labels
        acc = sess.run(accuarcy,feed_dict={x : xs,y_: ys})
        print(acc)


def main():
    mnist=input_data.read_data_sets("D:\\input_date.py", one_hot=True)
    test(mnist)

main()

        
