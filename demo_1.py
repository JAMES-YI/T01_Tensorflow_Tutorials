"""JYI, 11/13/2018 """ 
# load data set, data exploration 
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels 
with open('train-images-idx3-ubyte.gz','rb') as f: 
    train_x = extract_images(f) 
with open('train-labels-idx1-ubyte.gz','rb') as f: 
    train_y = extract_labels(f) 
with open('t10k-images-idx3-ubyte.gz','rb') as f: 
    test_x = extract_images(f) 
with open('t10k-labels-idx1-ubyte.gz','rb') as f: 
    test_y = extract_labels(f)
    
import matplotlib.pyplot as plt 
fig1 = plt.figure(1,figsize=(9,6)) 
plt.imshow(train_x[0].reshape((28,28))) 
fig1.suptitle('Training data sample',fontsize = 10) 
fig2 = plt.figure(2,figsize=(9,6)) 
plt.imshow(test_x[0].reshape((28,28))) 
fig2.suptitle('Testing data sample',fontsize = 10) 
plt.show()

print('train_y[0]:{}'.format(train_y[0])) # 5 
print('train_x.shape:{}'.format(train_x.shape)) # (60000, 28, 28, 1) 
print('train_y.shape:{}'.format(train_y.shape)) # (60000,) 
print('test_x.shape:{}'.format(test_x.shape)) # (10000, 28, 28, 1) 
print('test_y.shape:{}'.format(test_y.shape)) # (10000,) 

# data set pre-processing 
import numpy as np 
num_class = 10 
num_feature = 784 
num_train = len(train_x) 
num_test = len(test_x) 
train_x = np.squeeze(train_x) 
train_x = train_x.reshape(num_train,num_feature) # (60000,784) 
train_x = train_x/255 
test_x = np.squeeze(test_x) 
test_x = test_x.reshape(num_test,num_feature) # (10000,784) 
test_x = test_x/255 
print('train_x.shape:{}'.format(train_x.shape)) 
print('test_x.shape:{}'.format(test_x.shape)) 

def label2vector(label):
    num_sample = len(label) 
    vector = np.zeros((num_sample,num_class)) 
    for i in range(0,num_sample): 
        # label 0 <-> position 1, label 9 <-> position 10 
        vector[i, label[i]] = 1 
    return vector

train_y = label2vector(train_y) 
test_y = label2vector(test_y) 
print('train_y.shape:{}'.format(train_y.shape)) 
print('test_y.shape:{}'.format(test_y.shape)) 
# construct neural network 

import tensorflow as tf 
batch_size = 100 
epochs = 100 
learning_rate = 0.01 
nn_config = [784,300,10] 
x_batch = tf.placeholder(tf.float32, shape=[None,num_feature], name='feature') 
y_batch = tf.placeholder(tf.float32, shape=[None,num_class], name='label') 

W1 = tf.Variable(tf.random_normal([nn_config[0],nn_config[1]], stddev=1), name='W1') 
b1 = tf.Variable(tf.random_normal([nn_config[1]],stddev=1), name='b1') 
W2 = tf.Variable(tf.random_normal([nn_config[1],nn_config[2]],stddev=1), name='W2') 
b2 = tf.Variable(tf.random_normal([nn_config[2]],stddev=1), name='b2') 

# forward propagation 
wsum1 = tf.add(tf.matmul(x_batch,W1), b1) 
aout1 = tf.nn.relu(wsum1) # sigmoid, tanh 
wsum2 = tf.add(tf.matmul(aout1, W2), b2) 
aout2 = tf.nn.softmax(wsum2) # (100,10) 
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_batch, logits = aout2) 
loss_batch = tf.reduce_sum(cross_entropy_loss) / batch_size 
y_label_pred = tf.argmax(aout2,axis=1)
y_label = tf.argmax(y_batch,axis=1)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) 
my_train = my_opt.minimize(cross_entropy_loss) 
init_op = tf.global_variables_initializer() 

import numpy as np 
def evaluate(label_true,label_pred): 
    return np.count_nonzero(label_true - label_pred) 

# training set up from sklearn.utils 

import shuffle 
with tf.Session() as sess: 
    sess.run(init_op) 
    num_batch = int(num_train / batch_size) 
    train_x_shuffle, train_y_shuffle = shuffle(train_x,train_y) 
    for epoch in range(epochs): 
        train_error_rate = [] 
        test_error_rate = [] 
        for i in range(num_batch): 
            ind_start = i*batch_size 
            ind_end = i*batch_size + batch_size 
            x_batch_val = train_x_shuffle[ind_start:ind_end] 
            y_batch_val = train_y_shuffle[ind_start:ind_end] 
            sess.run(my_train,
                     feed_dict={x_batch:x_batch_val, y_batch:y_batch_val}) 
            # _W1, _W2, _b1, _b2 = sess.run([W1,W2,b1,b2]) 
            # print("b2: {}".format(_b2)) 
            _loss_batch = sess.run([loss_batch], 
                                   feed_dict={x_batch:x_batch_val, y_batch:y_batch_val}) 
            y_label_train, y_label_pred_train = sess.run([y_label, y_label_pred], 
                                                         feed_dict={x_batch:x_batch_val, y_batch:y_batch_val}) 
            num_error_train = evaluate(np.array(y_label_train), np.array(y_label_pred_train)) 
            train_error_rate = float(num_error_train) / float(batch_size) 
            y_label_test, y_label_pred_test = sess.run([y_label,y_label_pred], 
                                                       feed_dict={x_batch:test_x, y_batch:test_y}) 
            num_error_test = evaluate(np.array(y_label_test), np.array(y_label_pred_test)) 
            test_error_rate = float(num_error_test) / float(len(test_y)) 
        
        print('status: {}/{} epochs, loss:{}'.format(epoch,epochs,_loss_batch)) 
        print('status: {}/{} epochs, train_error_rate:{}, test_error_rate:{}'.format(epoch, epochs, train_error_rate,test_error_rate))
    
    
    
    
    