"""
Code from https://github.com/gongzhitaao/tensorflow-adversarial

Use DeepFool to craft adversarials on MNIST.

Modified by JYI, 05/28/2019
- add comments
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from deepfool import deepfool


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


class Dummy:
    pass


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

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
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
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
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_deepfool(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate DeepFool by running env.xadv.
    """
    print('\nMaking adversarials via DeepFool')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.xadv, feed_dict={env.x: X_data[start:end],
                                            env.adv_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv

if __name__ == "__main__":
    img_size = 28
    img_chan = 1
    n_classes = 10


    print('\nLoading MNIST and preprocessing')
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
    X_train = X_train.astype(np.float32) / 255
    X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
    X_test = X_test.astype(np.float32) / 255
    to_categorical = tf.keras.utils.to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    ind = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[ind], y_train[ind]

    VALIDATION_SPLIT = 0.1
    n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
    X_valid = X_train[n:]
    X_train = X_train[:n]
    y_valid = y_train[n:]
    y_train = y_train[:n]

    print('\nConstruction of classifier')

    env = Dummy()

    '''JYI
    how to proceed if not using Dummy class
    why extra tf.variable_scope('acc')
    with tf.variable_scope('model', reuse=True)
    
    '''
    with tf.variable_scope('model'):
        env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                               name='x')
        env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
        env.training = tf.placeholder_with_default(False, (), name='mode')
        env.ybar, logits = model(env.x, logits=True, training=env.training)

        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                           logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')
        # print(env.loss) Tensor("model/loss:0", shape=(), dtype=float32)


        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer()
            env.train_op = optimizer.minimize(env.loss)

        env.saver = tf.train.Saver()

    with tf.variable_scope('model', reuse=True): # cannot be deleted
        env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
        env.xadv = deepfool(model, env.x, epochs=env.adv_epochs)
        # print(env.adv_epochs) Tensor("model_1/adv_epochs:0", shape=(), dtype=int32)


    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('\nTraining a classifier')

    train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=1,
          name='mnist')

    print('\nEvaluating the classifier on clean data')

    evaluate(sess, env, X_test, y_test)

    print('\nGenerating adversarial data using trained classifier')

    X_adv = make_deepfool(sess, env, X_test, epochs=1)

    print('\nEvaluating on adversarial data using trained classifier')

    evaluate(sess, env, X_adv, y_test)

    print('\nRandomly sample adversarial data from each category')

    y1 = predict(sess, env, X_test)
    y2 = predict(sess, env, X_adv)

    z0 = np.argmax(y_test, axis=1) # ground truth label
    z1 = np.argmax(y1, axis=1) # predicted label for clean data
    z2 = np.argmax(y2, axis=1) # predicted label for adversarial data

    # report results
    IllustrationNumber = 5
    fig = plt.figure(figsize=(IllustrationNumber, 2.2))
    gs = gridspec.GridSpec(2, IllustrationNumber, wspace=0.05, hspace=0.05)

    for i in range(IllustrationNumber):
        print('Target {0}'.format(i))
        ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
        ind = np.random.choice(ind)
        xcur = [X_test[ind], X_adv[ind]]
        ycur = y2[ind]
        zcur = z2[ind]

        for j in range(2):
            img = np.squeeze(xcur[j])
            ax = fig.add_subplot(gs[j, i])
            ax.imshow(img, cmap='gray', interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(zcur, ycur[zcur]), fontsize=12)

    print('\nSaving figure')
    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/deepfool_mnist.png')
