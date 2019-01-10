import tensorflow as tf

session = tf.Session()

LSTM_SIZE = 4  # 4 hidden nodes = state_dim = the output_dim
input_dim = 6
num_layers = 2

cells = []
for _ in range(num_layers):
    cell = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE)
    cells.append(cell)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells)

# RNN Creation
data = tf.placeholder(tf.float32, [None, None, input_dim])  # Batch size x time steps x features.
output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

# Batch size x time steps x features.
sample_input = [[[1, 2, 3, 4, 3, 2], [1, 2, 1, 1, 1, 2], [1, 2, 2, 2, 2, 2]],
                [[1, 2, 3, 4, 3, 2], [3, 2, 2, 1, 1, 2], [0, 0, 0, 0, 3, 2]]]

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(output, feed_dict={data: sample_input}))
