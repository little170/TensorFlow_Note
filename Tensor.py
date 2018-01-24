import tensorflow as tf

tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])

sess = tf.Session() # open a session

print ('Print Tensor:',tensor)
P_tensor = sess.run(tensor)
print ('Print after running:',P_tensor)

sess.close() # close session
