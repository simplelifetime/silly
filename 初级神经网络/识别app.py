import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
MNIST_data_folder="D:\\input_date.py"  
mnist=input_data.read_data_sets(MNIST_data_folder,one_hot=True)
def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28), Image.ANTIALIAS)   #Image.ANTIALIAS 为抗锯齿操作
	im_arr = np.array(reIm.convert('L'))
	threshold = 50
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
 			if (im_arr[i][j] < threshold):
 				im_arr[i][j] = 0
			else: 
				im_arr[i][j] = 255

	nm_arr = im_arr.reshape([1, 784])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0/255.0)

	return img_ready

def application():
	testNum = input("input the number of test pictures:")
	for i in range(testNum):
		testPic = raw_input("the path of test picture:")
		testPicArr = pre_pic(testPic)
		preValue = restore_model(testPicArr)
		print "The prediction number is:", preValue

def main():
	application()
