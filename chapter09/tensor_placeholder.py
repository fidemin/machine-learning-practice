
import tensorflow as tf

input1 = tf.placeholder(tf.float32, 3)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

session = tf.InteractiveSession()
print(session.run([output], feed_dict={input1:[1., 2., 3.], input2:[3.]}))
session.close()
