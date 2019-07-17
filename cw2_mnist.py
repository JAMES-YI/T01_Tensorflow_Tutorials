"""
Use CW method to craft adversarial on MNIST.

Note that instead of find the optimized image for each image, we do a batched
attack without binary search for the best possible solution.  Thus, the result
is worse than reported in the original paper.  To achieve the best result
requires more computation, as demonstrated in another example.

Code from: https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/attacks/cw.py#L105

modified by JYI, 04/20/2019
- add comments
modified by JYI, 05/24/2019
-
"""
import os
from timeit import default_timer
import numpy as np
import matplotlib
# matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from cw import cw

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def evaluate(env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = env.sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(env.sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            env.sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                                  env.y: y_data[start:end],
                                                  env.training: True})
        if X_valid is not None:
            evaluate(env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(env.sess, 'model/{}'.format(name))

def predict(env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = env.sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_cw(env, X_data, epochs=1, eps=0.1, batch_size=32):
    """
    Generate adversarial via CW optimization.
    For each batch of adversarial samples, update epochs iterations to generate adversarial samples
    1 env is a class, what should it contain?
    """
    print('\nMaking adversarials via CW')

    n_sample = X_data.shape[0]
    n_batch = int(n_sample / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        end = min(n_sample, (batch+1) * batch_size)
        start = end - batch_size
        feed_dict = {
            env.x_fixed: X_data[start:end],
            env.adv_eps: eps,
            env.adv_y: 1} 
        env.sess.run(env.noise.initializer)
        for epoch in range(epochs): # iterations are required
            env.sess.run(env.adv_train_op, feed_dict=feed_dict) # training to generate adversarial samples

        xadv = env.sess.run(env.xadv, feed_dict=feed_dict) # obtain adversarial samples
        X_adv[start:end] = xadv

    return X_adv


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy: # empty class whose contents can be filled during running
    pass

if __name__ == '__main__':

    # data load and preprocessing
    img_size = 28
    img_chan = 1
    n_classes = 10
    batch_size = 32

    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
    X_train = X_train.astype(np.float32) / 255
    X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
    X_test = X_test.astype(np.float32) / 255
    to_categorical = tf.keras.utils.to_categorical
    y_train = to_categorical(y_train) # vectorized label
    y_test = to_categorical(y_test)

    ind = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[ind], y_train[ind]

    VALIDATION_SPLIT = 0.1
    n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
    X_valid = X_train[n:]
    X_train = X_train[:n]
    y_valid = y_train[n:]
    y_train = y_train[:n]

    # construct computational graph
    '''JYI
    structure of env
    (1) env.x, env.y, env.ybar, env.acc, env.loss, env.train_op, env.saver
    (2) 
    
    The training, evaluation, and adversarial attacking are within one graph.
    
    '''
    env = Dummy()
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        
        # construct a classification model, placeholders, logit, probability distribution, loss, optimizer
        env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                               name='x')
        env.y = tf.placeholder(tf.float32, (None, n_classes), name='y') # one hot vector labels
        env.training = tf.placeholder_with_default(False, (), name='mode')

        env.ybar, logits = model(env.x, logits=True, training=env.training) # function call

        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                               beta1=0.9,beta2=0.999,
                                               epsilon=1e-08,use_locking=False,name='Adam')
            vs = tf.global_variables()
            print(f'tf.global_variables {vs}\n')
            env.train_op = optimizer.minimize(env.loss, var_list=vs) # you can specify the objective function and optimization variables

        env.saver = tf.train.Saver()

        # configurations for generating adversarial samples
        env.x_fixed = tf.placeholder(
            tf.float32, (batch_size, img_size, img_size, img_chan),
            name='x_fixed')
        env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps') # perturbation tolerance
        env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')

        optimizer = tf.train.AdamOptimizer(learning_rate=0.1,
                                           beta1=0.9,beta2=0.999,
                                           epsilon=1e-08,use_locking=False,name='Adam')
        env.adv_train_op, env.xadv, env.noise = cw(model, env.x_fixed,
                                                   y=env.adv_y, eps=env.adv_eps,
                                                   optimizer=optimizer)

    env.sess = tf.InteractiveSession()
    env.sess.run(tf.global_variables_initializer())
    env.sess.run(tf.local_variables_initializer())

    # train a classifier, attacking a classifier
    print('\nTraining and evaluating a classifier on clean data')
    print('\nlog of training classifier')
    train(env, X_train, y_train, X_valid, y_valid, load=False, epochs=1,
          name='mnist')
    print('Evaluating over un-attacked samples...\n')
    evaluate(env, X_test, y_test)

    n_sample = 128
    ind = np.random.choice(X_test.shape[0], size=n_sample, replace=False)
    X_test = X_test[ind]
    y_test = y_test[ind]
    X_adv = make_cw(env, X_test, eps=0.002, epochs=100) # env contains the graph needed for generating adversarial samples
    print('Evaluating over attacked samples...\n')
    evaluate(env, X_adv, y_test)
    
    # extract samples which are correctly classified before attacking, but wrongly classified after attacking
    y1 = predict(env, X_test)
    y2 = predict(env, X_adv)

    z0 = np.argmax(y_test, axis=1) # ground truth
    z1 = np.argmax(y1, axis=1) # predicted label for benign samples
    z2 = np.argmax(y2, axis=1) # predicted label for adversarial samples
    
    ind = np.logical_and(z0 == z1, z1 != z2) # correct for benign samples, wrong for adversarial samples
    X_test = X_test[ind]
    X_adv = X_adv[ind]
    z1 = z1[ind] # correct labels
    z2 = z2[ind] # wrong labels
    y2 = y2[ind] # contain the confidence of a particular adversarial sample for each class
    ind = np.sum(ind)

    # Illustrations with examples
    IllustrationNumber = 5
    cur = np.random.choice(ind, size=IllustrationNumber)

    X_org = np.squeeze(X_test[cur])
    X_tmp = np.squeeze(X_adv[cur])
    y_tmp = y2[cur] # confidence

    fig = plt.figure()
    gs = gridspec.GridSpec(3, IllustrationNumber+1, width_ratios=[1]*IllustrationNumber + [0.05],
                          wspace=0.01, hspace=0.01)
    '''JYI
    1 the width of the 6 figures has a ratio, 1:1:1:1:1:0.05
    2 plot figures column by column
    '''

    label = np.argmax(y_tmp, axis=1) # adversarial labels
    proba = np.max(y_tmp, axis=1) # probability for assigning adversarial labels

    for i in range(IllustrationNumber):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(X_org[i], cmap='gray', interpolation='none')
        ax.set_xlabel('Orig')

        ax = fig.add_subplot(gs[1, i])
        img = ax.imshow(X_tmp[i]-X_org[i], cmap='RdBu_r', vmin=-1,
                        vmax=1, interpolation='none')

        ax = fig.add_subplot(gs[2, i])
        ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
        ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]), fontsize=12)

    # figure format set up, put a color bar for the whole image
    ax = fig.add_subplot(gs[1, IllustrationNumber])
    dummy = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-1,
                                                                    vmax=1))
    dummy.set_array([])
    fig.colorbar(mappable=dummy, cax=ax, ticks=[-1, 0, 1], ticklocation='right')

    print('\nSaving figure')

    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/cw2_illustration.pdf')
