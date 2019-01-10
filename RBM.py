import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from PIL import Image

from utils import tile_raster_images

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

vb = tf.placeholder("float", [784])
hb = tf.placeholder("float", [500])

W = tf.placeholder("float", [784, 500])

# FORWARD PASS
X = tf.placeholder("float", [None, 784])
_h0 = tf.nn.sigmoid(tf.matmul(X, W) + hb)  # probabilities of the hidden units
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # sample_h_given_X

# Example of Sampling
with tf.Session() as session:
    a = tf.constant([0.7, 0.1, 0.8, 0.2])
    print(session.run(a))
    b = session.run(tf.random_uniform(tf.shape(a)))
    print(b)
    print(session.run(a - b))
    print (session.run(tf.sign(a - b)))
    print (session.run(tf.nn.relu(tf.sign(a - b))))

# BACKWARD PASS
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))  # sample_v_given_h
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

# Calculate gradients
alpha = 1.0  # learning-rate
w_pos_grad = tf.matmul(tf.transpose(X), h0)  # Positive gradient (Reconstruction in the first pass)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)  # Negaive gradient (Reconstruction1)
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(X)[0])  # Contrastive divergence
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# tf.reduce_mean computes the mean of elements across dimensions of a tensor
err = tf.reduce_mean(tf.square(X - v1))

cur_w = np.zeros([784, 500], np.float32)
cur_vb = np.zeros([784], np.float32)
cur_hb = np.zeros([500], np.float32)
prv_w = np.zeros([784, 500], np.float32)
prv_vb = np.zeros([784], np.float32)
prv_hb = np.zeros([500], np.float32)

init = tf.global_variables_initializer()

#Parameters
epochs = 5
batchsize = 100
weights = []
errors = []

with tf.Session() as session:
    session.run(init)
    session.run(err, feed_dict={X: trX, W: prv_w, vb: prv_vb, hb: prv_hb})
    for epoch in range(epochs):
        for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
            batch = trX[start:end]
            cur_w = session.run(update_w, feed_dict={ X: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_vb = session.run(update_vb, feed_dict={  X: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_hb = session.run(update_hb, feed_dict={ X: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            prv_w = cur_w
            prv_vb = cur_vb
            prv_hb = cur_hb
            if start % 10000 == 0:
                errors.append(session.run(err, feed_dict={X: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
                weights.append(cur_w)
        print ('Epoch: %d' % epoch,'reconstruction error: %f' % errors[-1])
