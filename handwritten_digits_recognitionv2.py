import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Start interactive session
session = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# initial parameters
width = 28
height = 28
flat = width * height  # number of pixels in one image
class_output = 10  # number of possible classifications for the problem

# Placeholders for inputs and outputs
x = tf.placeholder(tf.float32, [None, flat])
y_ = tf.placeholder(tf.float32, [None, class_output])

# Converting images into tensors
x_image = tf.reshape(x, [-1, 28, 28, 1])

# ---------------------------------------- CONVOLUTIONAL LAYER 1 ----------------------------------------
# Kernel definition of shape [height, width, in_channels, out_channels]
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))  # 32 different filters are applied on each image
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  # 32 biases for 32 outputs

convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
# Apply the activation function
h_conv1 = tf.nn.relu(convolve1)
# Apply max_pooling
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max_pool_2x2

# ---------------------------------------- CONVOLUTIONAL LAYER 2 ----------------------------------------
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))  # 64 different filters are applied on each image
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  # 64 biases for 64 outputs

convolve2 = tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
h_conv2 = tf.nn.relu(convolve2)
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max_pool_2x2

# ---------------------------------------- FULLY CONNECTED LAYER ----------------------------------------
# Flattening second layer
layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

fc1 = tf.matmul(layer2_matrix, W_fc1) + b_fc1

# Apply ReLu activation function
h_fc1 = tf.nn.relu(fc1)

# -------------------------------------------- DROPOUT LAYER --------------------------------------------
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)

# -------------------------------------------- READOUT LAYER --------------------------------------------
# SoftMax Fully Connected Layer
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))  # 1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))  # 10 possible digits

fc = tf.matmul(layer_drop, W_fc2) + b_fc2

# Apply SoftMax activation function
y_CNN = tf.nn.softmax(fc)

# Loss Function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))
# Define Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())

for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

