import tensorflow as tf

log_path = "/tmp/logs/"

with name_scope():
     x = tf.placeholder(tf.float32,name='x')
     weight = tf.placeholder(tf.float32,name='weight')
     bias = tf.constant(6.0)

     y = tf.add(x*weight,bias) 

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
     sess.run(init_op)
     writer = tf.summary.FileWriter(log_path,graph=sess.graph)
     result = sess.run(y,feed_dict={x:3.0,weight:2.0})

     print(result)

