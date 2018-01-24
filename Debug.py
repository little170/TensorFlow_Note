import tensorflow as tf
from tensorflow.python import debug as tf_debug

x = tf.placeholder(tf.float32,name='x')
weight = tf.placeholder(tf.float32,name='weight')
bias = tf.constant(6.0)

y = tf.add(tf.multiply(x,weight),bias) 

with tf.Session() as sess:

     debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
     result = debug_sess.run(y,feed_dict={x:3.0,weight:2.0})
     print(result)

