import tensorflow as tf

import reader

# Initial weight scale
init_scale = 0.1
# Initial learning rate
learning_rate = 1.0
# Maximum permissible norm for the gradient (For gradient clipping -- another measure against Exploding Gradients)
max_grad_norm = 5
# The number of layers in our model
num_layers = 2
# The total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"
num_steps = 20
# The number of processing units (neurons) in the hidden layers
hidden_size = 200
# The maximum number of epochs trained with the initial learning rate
max_epoch = 4
# The total number of epochs in training
max_max_epoch = 13
# The probability for keeping data in the Dropout Layer (This is an optimization,
# but is outside our scope for this notebook!)
# At 1, we ignore the Dropout Layer wrapping.
keep_prob = 1
# The decay for the learning rate
decay = 0.5
# The size for each batch of data
batch_size = 30
# The size of our vocabulary
vocab_size = 10000
# Training flag to separate training from testing
is_training = 1
# Data directory for our dataset
data_dir = "data/simple-examples/data/"

session = tf.InteractiveSession()

# Reads the data and separates it into training data, validation data and testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _ = raw_data

itera = reader.ptb_iterator(train_data, batch_size, num_steps)
first_touple = itera.__next__()
x = first_touple[0]
y = first_touple[1]

size = hidden_size

_input_data = tf.placeholder(tf.int32, [batch_size, num_steps])  # [30#20]
_targets = tf.placeholder(tf.int32, [batch_size, num_steps])  # [30#20]

feed_dict = {_input_data: x, _targets: y}

session.run(_input_data, feed_dict)

lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=0.0)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

_initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

session.run(_initial_state, feed_dict)

embedding = tf.get_variable("embedding", [vocab_size, hidden_size])  # [10000x200]
session.run(tf.global_variables_initializer())
session.run(embedding, feed_dict)

# Define where to get the data for our embeddings from
inputs = tf.nn.embedding_lookup(embedding, _input_data)  # shape=(30, 20, 200)

session.run(inputs[0], feed_dict)
