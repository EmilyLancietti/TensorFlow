import tensorflow as tf

# Building graph

# 10x10 image (4D Tensor (batch_size, width, height, number of channels)
# 3x3 filter (4D Tensor (width, height, channels, n_ilters)

input = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))

# the output size will be the same as the input
# strides determines how much the window shifts by in each of the dimensions
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  # without zero-padding mode

# the output size will be input_size - kernel_dimension + 1 = 10 - 3 + 1 = 8 => 8x8
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')  # zero-padding mode

# Initialization and session
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    print('Input \n')
    print('{0} \n'.format(input.eval()))
    print('Filter \n')
    print('{0} \n'.format(filter.eval()))
    print('Result/Feature Map with valid positions \n')
    result = session.run(op)
    print(result)
    print('Result/Feature Map with padding \n')
    result = session.run(op2)
    print(result)
