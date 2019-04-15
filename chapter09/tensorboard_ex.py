
import tensorflow as tf

session = tf.InteractiveSession()
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b)
writer = tf.summary.FileWriter('./graphs', session.graph)
session.run(x)
writer.close()
session.close()
