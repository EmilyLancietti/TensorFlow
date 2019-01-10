import numpy as np
import tensorflow as tf

session = tf.Session()

LSTM_CELL_SIZE = 4 # number of units

lstm_cell = tf.nn.rnn_cell.LSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
# state: tuple of 2 elements of size 1x4, one for passing prv_output to next time step
# and another for passing the prv_state to the next time step
state = (tf.zeros([2, LSTM_CELL_SIZE]),)*2

# Sample input
sample_input = tf.constant([[1, 2, 3, 4, 3, 2], [3, 2, 2, 2, 2, 2]], dtype=tf.float32)
print(session.run(sample_input))

with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
session.run(tf.global_variables_initializer())

# the states have 2 parts:
# c: the new state
# h: the output
print (session.run(state_new))
print (session.run(output))

session.close()
