import tensorflow as tf

# Source operations: do non require any input
a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a, b)  # c = a + b works as well

# session = tf.Session()
# result = session.run(c)
# print(result)
# session.close() # close session to release resources

# The session will close automatically using a with block
with tf.Session() as session:
    result = session.run(c)
    print(result)
