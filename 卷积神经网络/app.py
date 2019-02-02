#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import forward1
import matplotlib.pyplot as plt
import os
model_path ="./modelCNN1/"
model_name="mnist_model"

def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, 784])
		keep_prob=tf.placeholder(tf.float32)
		lamb=tf.placeholder(tf.float32)
		y = forward1.forward(x,keep_prob,lamb)
		preValue =y
		saver = tf.train.Saver()

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(model_path)
			# saver.restore(sess, ckpt.model_checkpoint_path)
			# preValue = sess.run(preValue, feed_dict={x:testPicArr})
			# return preValue
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				preValue = sess.run(preValue, feed_dict={x:testPicArr,keep_prob:1.0,lamb:0.0})
				return preValue

			else:
				print("No checkpoint file found")
				return -1

def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28), Image.ANTIALIAS)
	im_arr = np.array(reIm.convert('L'))
	threshold = 50
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
			if (im_arr[i][j] < threshold):
				im_arr[i][j] = 0
			else: im_arr[i][j] = 255

	nm_arr = im_arr.reshape([1, 784])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0/255.0)
	plt.imshow(im_arr)
	plt.show()
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