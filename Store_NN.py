from __future__ import print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# training on MNIST but only on digits 0 to 4
X_train1 = mnist.train.images[mnist.train.labels < 5]
y_train1 = mnist.train.labels[mnist.train.labels < 5]
X_test1 = mnist.test.images[mnist.test.labels < 5]
y_test1 = mnist.test.labels[mnist.test.labels < 5]
# One-hot Encoding Label
y_train1 = tf.one_hot(indices=y_train1, depth=5, on_value=1.0, off_value=0.0, axis=-1)
y_test1 = tf.one_hot(indices=y_test1, depth=5, on_value=1.0, off_value=0.0, axis=-1)

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_epoch = 1
logs_path = '/tmp/Store/logs'
Input_Node = 784
Output_Node = 5
Layer1_Node = 10
Layer2_Node = 10
Layer3_Node = 10

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, Input_Node], name='InputData')
# 0-4 digits recognition => 5 classes
y = tf.placeholder(tf.float32, [None, Output_Node], name='LabelData')

# Set model weights

w1 = tf.Variable(tf.truncated_normal([Input_Node,Layer1_Node], stddev=0.1), name='Weights1')
b1 = tf.Variable(tf.constant(0.1,shape=[Layer1_Node]), name='Bias1')

w2 = tf.Variable(tf.truncated_normal([Layer1_Node,Layer2_Node], stddev=0.1), name='Weights2')
b2 = tf.Variable(tf.constant(0.1,shape=[Layer2_Node]), name='Bias2')

w3 = tf.Variable(tf.truncated_normal([Layer2_Node, Output_Node], stddev=0.1), name='Weights3')
b3 = tf.Variable(tf.constant(0.1,shape=[Output_Node]), name='Bias3')

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
# Model
h1 = tf.nn.relu(tf.matmul(x,w1) + b1, name='h1')
h2 = tf.nn.relu(tf.matmul(h1,w2) + b2, name='h2')
logits =tf.add (tf.matmul(h2,w3),b3,name='logits')
pred = tf.nn.softmax(tf.matmul(h2,w3) + b3) # Softmax
# Minimize error using cross entropy
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y),name='loss')
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
# Accuracy
acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
mean_acc = tf.reduce_mean(tf.cast(acc, tf.float32),name='accuracy')

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cross_entropy)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", mean_acc)
# Create a summary to monitor weight
tf.summary.histogram("weight",w1)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Declare a tf.train.Saver to save model
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    # One-hot encoding Labels
    y_train = sess.run(y_train1)
    y_test = sess.run(y_test1)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for batch in range(total_batch):
            if batch == total_batch - 1:
                a, c, summary = sess.run([optimizer, cross_entropy,  merged_summary_op],
                                feed_dict={x: X_train1[batch*batch_size:], y: y_train[batch*batch_size:]})
            else:
                a, c, summary = sess.run([optimizer, cross_entropy, merged_summary_op],
                         feed_dict={x: X_train1[batch*batch_size : (batch+1)*batch_size ],
                                    y: y_train[batch*batch_size : (batch+1)*batch_size]})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + batch)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            
    print("Optimization Finished!")

    #Save the testing results of each stage
    save_path = saver.save(sess, '/tmp/model/model.ckpt')
    # Test model
    # Calculate accuracy
    print("Accuracy:", mean_acc.eval({x: X_test1, y: y_test}))

print("\n")
print("Model Save Path=",save_path)
print("Run the command line:\n" \
          "-->$ tensorboard --host=< IP address> --logdir=<logs path> --port=<port>" \
          "\nThen open http://0.0.0.0:6006/ into your web browser")

