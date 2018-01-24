# TensorFlow_Note
## Install Tensorflow
(take MAC as example)
1. install virtualenv
```Clik=
$ pip install --upgrade virtualenv 
```
2. Create a virtualenv environment
```Clik=+
$ virtualenv --system-site-packages <targetDirectory> # for Python 2.7
$ virtualenv --system-site-packages -p python3 <targetDirectory> # for Python 3.n
```
3. Activate the virtualenv environment
```Clik=+
$ source <targetDirectory>/bin/activate 
(tensorflow)$
```
4. Ensure pip ≥8.1 is installed
```Clik=+
(tensorflow)$ easy_install -U pip
```
5. Install TensorFlow and all the packages that TensorFlow requires
```Clik=+
(tensorflow)$ pip install --upgrade tensorflow # for Python 2.7
(tensorflow)$ pip3 install --upgrade tensorflow # for Python 3.n
```
## 

Overview of Tensorflow:

使用 graph 来表示計算任務.
在被稱之為 Session 的context 中執行graph.
使用 tensor 表示數據.
通過 Variable 維護狀態.
使用 feed 和 fetch 可以做arbitrary operation 賦值或者從其中獲取數據.
## What is Tensor?
* Tensorflow is a framework to define and run computations involving tensors.
* A tensor is a generalization of vectors and matrices to potentially higher dimensions. 
```
# example
tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])
print tensor

with tf.Session() as sess:
     cont = sess.run(tensor)
     print cont
```
## How to use Session?
```Clik=
# session用法一
sess = tf.Session()
result = sess.run(product)
print result
sess.close()

## session用法二，不用考慮close，會自動關閉

with tf.Session() as sess:
    result = sess.run(product)
    print result
```
## Feed Data
```
x = tf.placeholder(tf.float32)
a = tf.constant(1.2)
y = tf.add(x,a)
with tf.Session() as sess:
     sess.run(y, feed_dict={x=3.2})
```
## Try plotting Dataflow Grpah
![](http://volibear.cs.nthu.edu.tw:3000/uploads/upload_d989c6cd61b311f6b709235cdf1fc29a.png)
1. Add Line 3 and Line 7 then execute .py
```Clik=
log_path = "/tmp/logs/"

with name_scope():
     #build your graph
     
with tf.Secssion() as sess:
     writer = tf.summary.FileWriter(log_path,graph=sess.graph)
```
2. Open another Terminal and executed
```
$ tensorboard --logdir=/tmp/logs
```
3. access localhost:6006


## Debugger
```Clik=
from tensorflow.python import debug as tfdbg
...
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    debug_sess.run(train_op) 
```
![](http://volibear.cs.nthu.edu.tw:3000/uploads/upload_dbbdc401512d50fc8aea528be7300388.png)
