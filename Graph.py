import tensorflow as tf

log_path = "/tmp/logs/"

with tf.name_scope(None) as scope:
     x = tf.placeholder(tf.float32,name='x')
     weight = tf.placeholder(tf.float32,name='weight')
     bias = tf.constant(6.0)

     y = tf.add(x*weight,bias) 


with tf.Session() as sess:

     writer = tf.summary.FileWriter(log_path,graph=sess.graph)
     result = sess.run(y,feed_dict={x:3.0,weight:2.0})

     print(result)

