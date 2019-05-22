import tensorflow as tf

W1 = tf.zeros((3,2)) # 상수 텐서
W2 = tf.Variable(tf.zeros((2,2)), name='weights')


session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

print(W1.eval())
print(W2.eval())

session.close()
