import tensorflow as tf

x = tf.placeholder(tf.float32,name='x')
weight = tf.placeholder(tf.float32,name='weight')
bias = tf.constant(6.0)

y = tf.add(tf.multiply(x,weight),bias) 

with tf.Session() as sess:
     result = sess.run(y,feed_dict={x:3.0,weight:2.0})
     print(result)

