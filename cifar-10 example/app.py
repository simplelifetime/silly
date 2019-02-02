import tensorflow as tf
import os
import backward,forward
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
data_dir="D:\python2\cifar-10\cifar-10-batches-py"
save_path="./modelcifar10/"
model_name="cifar-10_model"
def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None,32,32,3])
		y = forward1.forward(x,0.0)
		preValue = tf.argmax(y,1)
		saver = tf.train.Saver()
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(save_path)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue

			else:
				print("No checkpoint file found")
				return -1

def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((32,32), Image.ANTIALIAS)
	im_arr = np.array(reIm.convert('P'))
	plt.imshow(im_arr)
	plt.show()
	nm_arr = im_arr.reshape([1,32,32,3])
	nm_arr_ = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr_, 1.0/255.0)
	return img_ready

def application():
	testNum = int(input("input the number of test pictures:"))
	for i in range(testNum):
		testPic = input("the path of test picture:")
		testPicArr = pre_pic(testPic)
		preValue = restore_model(testPicArr)
		print("The prediction number is:", preValue)

def main():
	application()

if __name__ == '__main__':
	main()	