import tensorflow as tf

# Retrieving MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

session = tf.InteractiveSession()

# Each input has 784 pixels distributed by a 28 width x 28 height matrix
# The 'shape' argument defines the tensor size by its dimensions.
# 1st dimension = None. Indicates that the batch size, can be of any size.
# 2nd dimension = 784. Indicates the number of pixels on a single flattened MNIST image.
x = tf.placeholder(tf.float32, shape=[None, 784])

# 10 possible classes (0,1,2,3,4,5,6,7,8,9)
# The 'shape' argument defines the tensor size by its dimensions.
# 1st dimension = None. Indicates that the batch size, can be of any size.
# 2nd dimension = 10. Indicates the number of targets/outcomes
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Creation of weights and biases initialized to zero
# Weight Tensor
W = tf.Variable(tf.zeros([784, 10]), tf.float32)
# Bias Tensor
b = tf.Variable(tf.zeros([10]), tf.float32)

# Variable Initialization
session.run(tf.global_variables_initializer())

# apply activation function
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Load 50 training examples for each training iteration
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

session.close()