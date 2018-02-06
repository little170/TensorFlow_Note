import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug
mnist = input_data.read_data_sets("MNIST_data/")

# training on MNIST but only on digits 0 to 4
X_train1 = mnist.train.images[mnist.train.labels >= 5]
y_train1 = mnist.train.labels[mnist.train.labels >= 5]-5
X_test1 = mnist.test.images[mnist.test.labels >= 5]
y_test1 = mnist.test.labels[mnist.test.labels >= 5]-5

# Import Pre-train Graph
saver = tf.train.import_meta_graph('/tmp/model/model.ckpt.meta')

graph = tf.get_default_graph()
# One-hot Encoding Label
y_train1 = tf.one_hot(indices=y_train1, depth=5, on_value=1.0, off_value=0.0, axis=-1)
y_test1 = tf.one_hot(indices=y_test1, depth=5, on_value=1.0, off_value=0.0, axis=-1)

# Parameters
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_epoch = 1
logs_path = '/tmp/transfer/logs'

# Get Tensor by name from Pre-train graph
x = graph.get_tensor_by_name("InputData:0")
y = graph.get_tensor_by_name("LabelData:0")
#loss = tf.get_default_graph().get_tensor_by_name("loss:0")
h2 = graph.get_tensor_by_name("h2:0")

w3 = tf.Variable(tf.truncated_normal([10, 5], stddev=0.1), name='W3')
b3 = tf.Variable(tf.constant(0.1,shape=[5]), name='B3')
logits =tf.add (tf.matmul(h2,w3),b3,name='logits')
pred = tf.nn.softmax(logits)
# Minimize error using cross entropy
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,var_list=[w3,b3])

# Accuracy
acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
mean_acc = tf.reduce_mean(tf.cast(acc, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # restore variables 
    saver.restore(sess, '/tmp/model/./model.ckpt')

    y_train = sess.run(y_train1)
    y_test = sess.run(y_test1)

    X_train = sess.run(h2,feed_dict={x:X_train1, y:y_train})
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for batch in range(total_batch):
            if batch == total_batch - 1:
                a, c = sess.run([optimizer, loss],
                                feed_dict={h2: X_train[batch*batch_size:], y: y_train[batch*batch_size:]})
            else:
                a, c = sess.run([optimizer, loss],
                         feed_dict={h2: X_train[batch*batch_size : (batch+1)*batch_size ],
                                    y: y_train[batch*batch_size : (batch+1)*batch_size]})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Transfer Learning")
    print("Using the model train in label 0-4")
    print("Test in label 5-9")
    print("acc:", sess.run(mean_acc,feed_dict={x: X_test1, y: y_test}))


