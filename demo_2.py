"""JYI
This file is to demonstrate the performance of cnn in image classification for MNIST data set

by JYI, 12/07/2018
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# In[]
sess = tf.Session()

# Load data
data_dir = 'HW6_data'
mnist = read_data_sets(data_dir)
train_feat = [0]*55000
test_feat = [0]*10000
for samp_idx,samp in enumerate(mnist.train.images):
    train_feat[samp_idx] = samp.reshape((28,28))
for samp_idx,samp in enumerate(mnist.test.images):
    test_feat[samp_idx] = samp.reshape((28,28))
train_feat = np.array(train_feat)
test_feat = np.array(test_feat)
train_feat
    
train_lab = mnist.train.labels # (55000,)
test_lab = mnist.test.labels # (10000,)

for i in np.arange(5):
    # plt.imshow(np.transpose(train_feat[i],[1,0]))
    plt.imshow(train_feat[i])
    plt.show()
# In[]: parameter setup
batch_size = 100
learn_rate = 0.005
I_w = train_feat.shape[1]
I_h = train_feat.shape[2]
N_class = 10
generations = 500
conv_fnum = 25 # two conv layers with same number of filters
conv_fsize = 4 
maxpool_size = 2 # two maxpool layers with same size 
fc = 100
test_eval_every = 50

# In[]
feat = tf.placeholder(tf.float32, shape=[batch_size,I_w,I_h,1])
lab = tf.placeholder(tf.int32, shape=(batch_size))

# Declare model parameters
conv1_weight = tf.Variable(tf.truncated_normal([conv_fsize, conv_fsize, 1, conv_fnum],
                                               stddev=0.1, dtype=tf.float32)) # 
conv1_bias = tf.Variable(tf.zeros([conv_fnum], dtype=tf.float32)) #

conv1_wsum = tf.nn.conv2d(feat, conv1_weight, strides=[1,1,1,1], padding='SAME') # 
conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1_wsum, conv1_bias)) # 
maxpool1 = tf.nn.max_pool(conv1_relu, ksize=[1,maxpool_size,maxpool_size,1],
                                strides=[1,maxpool_size,maxpool_size,1], padding='SAME') # 

conv2_weight = tf.Variable(tf.truncated_normal([conv_fsize, conv_fsize, conv_fnum, conv_fnum],
                                               stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv_fnum], dtype=tf.float32))

conv2_wsum = tf.nn.conv2d(maxpool1, conv2_weight, strides=[1,1,1,1], padding='SAME')
conv2_relu = tf.nn.relu(tf.nn.bias_add(conv2_wsum, conv2_bias))
maxpool2 = tf.nn.max_pool(conv2_relu, ksize=[1,maxpool_size,maxpool_size,1],
                                strides=[1,maxpool_size,maxpool_size,1], padding='SAME')

conv_output = maxpool2
conv_shape = conv_output.get_shape().as_list()
fc_Isize = conv_shape[1]*conv_shape[2]*conv_shape[3]
conv_output_flat = tf.reshape(conv_output, [batch_size,fc_Isize])

fc1_weight = tf.Variable(tf.truncated_normal([fc_Isize, fc], stddev=0.1, dtype=tf.float32)) # 
fc1_bias = tf.Variable(tf.truncated_normal([fc], stddev=0.1, dtype=tf.float32))
fc1_relu = tf.nn.relu(tf.add(tf.matmul(conv_output_flat, fc1_weight), fc1_bias))

fc2_weight = tf.Variable(tf.truncated_normal([fc, N_class], stddev=0.1, dtype=tf.float32))
fc2_bias = tf.Variable(tf.truncated_normal([N_class], stddev=0.1, dtype=tf.float32))
fc_wsum = tf.add(tf.matmul(fc1_relu,fc2_weight), fc2_bias)

# In[]: evaluation metric
nn_output = fc_wsum

# Declare Loss Function (softmax cross entropy)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nn_output, labels=lab)) #
prediction = tf.nn.softmax(nn_output)

# In[]
# Create an optimizer
my_optimizer = tf.train.MomentumOptimizer(learn_rate, 0.9) # 
train_update = my_optimizer.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Start training loop
train_loss = []
train_acc = []
test_acc = []
for i in range(generations):
    batch_ind = np.random.choice(len(train_feat), size=batch_size) # 
    feat_val = train_feat[batch_ind] # (100,28,28)
    feat_val = np.expand_dims(feat_val, 3) # (100,28,28,1)
    lab_val = train_lab[batch_ind]
    dict_val = {feat: feat_val, lab: lab_val}
    
    sess.run(train_update, feed_dict=dict_val)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=dict_val)
    lab_pred = np.argmax(temp_train_preds,axis=1)
    temp_train_acc = np.sum(lab_pred==lab_val) / float(batch_size)
    
    
    if (i+1) % test_eval_every == 0:
        test_batch_ind = np.random.choice(len(test_feat), size=batch_size)
        test_feat_val = test_feat[test_batch_ind]
        test_feat_val = np.expand_dims(test_feat_val, 3)
        test_lab_val = test_lab[test_batch_ind]
        test_dict_val = {feat: test_feat_val, lab: test_lab_val}
        
        test_pred = sess.run(prediction, feed_dict=test_dict_val)
        test_lab_pred = np.argmax(test_pred,axis=1)
        temp_test_acc = np.sum(test_lab_pred==test_lab_val) / float(batch_size)
        
        # Record and print results
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))


# In[]    
# Matlotlib code to plot the loss and accuracies
eval_ind = range(0, generations, test_eval_every)
# Plot loss over time
plt.plot(eval_ind, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot train and test accuracy
plt.plot(eval_ind, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_ind, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()