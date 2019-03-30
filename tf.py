import tensorflow as tf
import numpy as np

w=tf.Variable(0,dtype=tf.float32)
cost=tf.add(tf.add((w**2),tf.multiply(-10.,w)),25)

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(20000):
	sess.run(train)

print(sess.run(w))
