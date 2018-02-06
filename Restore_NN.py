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
training_epochs = 20
batch_size = 100
display_epoch = 1
logs_path = '/tmp/transfer/logs'

# Get Tensor by name from Pre-train graph
x = graph.get_tensor_by_name("InputData:0")
y = graph.get_tensor_by_name("LabelData:0")

h1 = graph.get_tensor_by_name("h1:0")
loss = graph.get_tensor_by_name("loss:0")
acc = graph.get_tensor_by_name("accuracy:0")

with tf.Session() as sess:
    # restore variables 
    saver.restore(sess, '/tmp/model/./model.ckpt')

    y_train = sess.run(y_train1)
    y_test = sess.run(y_test1)

    # Test model
    # Calculate accuracy
    print("Restore model without training")
    print("Using the model train in label 0-4")
    print("Test in label 5-9")
    print("acc:", sess.run(acc,feed_dict={x: X_test1, y: y_test}))


