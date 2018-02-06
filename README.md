## 初步認識 Tensorflow
**1. Tensor.py**\
認識Tensor

**2. Feed.py**\
Feed 資料

**3. Graph.py**\
利用Tensorboard畫出Graph

**4. Debug.py**\
用Debugger得知各Tensor的值

**5. NN.py**\
Debugger with Neural Network，試著熟悉debugger觀看各Tensor的值\
structure: 784 node in input layer, 10 node in 1 hidden layer, 10 node in output layer

## 儲存、恢復、遷移訓練(Transfer Learning)模型

**1. Simple_store.py & Simple_restore.py**\
簡單的儲存和恢復模型範例

**2. NN.py** Neural Network範例

**3. Store_NN.py & Restore_NN.py & Transfer_NN.py** NN儲存、恢復、遷移訓練範例\
structure: 784 node in input layer, 10 node in 3 hidden layer, 5 node in output layer\
利用sklearn之手寫辨識Dataset\
Store_NN.py使用Dataset中label 0-4之資料做training\
Restore_NN.py使用Dataset中label 5-9之資料及Store_NN.py之模型做測試\
Transfer_NN.py使用Dataset中label 5-9之資料及Store_NN.py之模型做測試，且重新train output layer的 weights
