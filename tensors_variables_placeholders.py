import tensorflow as tf

# ------------------------------ TENSORS ------------------------------
# Define multidimensional arrays
Scalar = tf.constant([2])
Vector = tf.constant([2, 3, 4])
Matrix = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
Tensor = tf.constant(
    [[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
     [[4, 5, 6], [5, 6, 7], [6, 7, 8]],
     [[7, 8, 9], [8, 9, 10], [9, 10, 11]]])

with tf.Session() as session:
    result = session.run(Scalar)
    print('Scalar (1 entry):\n %s \n' % result)
    result = session.run(Vector)
    print('Vector (3 entries):\n %s \n' % result)
    result = session.run(Matrix)
    print('Matrix (3x3 entries):\n %s \n' % result)
    result = session.run(Tensor)
    print('Tensor (3x3x3 entries):\n %s \n' % result)

# ----------------------------- VARIABLES -----------------------------
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables need to be initialized before a graph can be run
# init_op = tf.initialize_all_variables()   deprecated
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    print(session.run(state))
    for _ in range(3):
        session.run(update)
        print(session.run(state))

# ----------------------------- PLACEHOLDERS -----------------------------
a = tf.placeholder(tf.float32)
b = a * 2

with tf.Session() as session:
    result = session.run(b, feed_dict={a: 3.5})
    print(result)
    result = session.run(b, feed_dict={a: [[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                                           [[4, 5, 6], [5, 6, 7], [6, 7, 8]],
                                           [[7, 8, 9], [8, 9, 10], [9, 10, 11]]]})
    print(result)
