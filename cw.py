import tensorflow as tf


__all__ = ['cw']

'''
Code from https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/attacks/cw.py#L105

Modified by JYI, 05/28/2019
- add comments
- suggest to read the paper about cw attack
- incoraporate cw.py to your own packages

'''


def cw(model, x, y=None, eps=1.0, T=2,
       optimizer=tf.train.AdamOptimizer(learning_rate=0.1), alpha=0.9,
       min_prob=0, clip=(0.0, 1.0)):
    """CarliniWagner (CW) attack.
    The idea of CW attack is to minimize a loss that comprises two parts: a)
    the p-norm distance between the original image and the adversarial image,
    and b) a term that encourages the incorrect classification of the
    adversarial images.

    :param model: The model wrapper.
    :param x: The input clean sample, usually a placeholder.  NOTE that the
              shape of x MUST be static, i.e., fixed when constructing the
              graph.  This is because there are some variables that depends
              upon this shape. (batch_size, img_size, img_size, img_chan)
    :param y: The target label.  Set to be the least-likely label when None.
    :param eps: The scaling factor for the second penalty term.
    :param T: The temperature for sigmoid function.  In the original paper,
              the author used (tanh(x)+1)/2 = sigmoid(2x), i.e., t=2.  During
              our experiment, we found that this parameter also affects the
              quality of generated adversarial samples.
    :param optimizer: The optimizer used to minimize the CW loss.  Default to
        be tf.AdamOptimizer with learning rate 0.1. Note the learning rate is
        much larger than normal learning rate.
    :param alpha: Used only in CW-L0.  The decreasing factor for the upper
        bound of noise.
    :param min_prob: The minimum confidence of adversarial examples.
        Generally larger min_prob wil lresult in more noise.
    :param clip: A tuple (clip_min, clip_max), which denotes the range of
        values in x.
    """

    '''JYI
    1 iterations are required to find adversarial samples
    2 here only performs one iteration
    3 motivation for xinv = tf.log(z / (1 - z)) / T 
    4 motivation for xadv = tf.sigmoid(T * (xinv + noise)) 
    5 motivation for xadv = xadv * (clip[1] - clip[0]) + clip[0]
    6 motivation for x_scaled = (x - clip[0]) / (clip[1] - clip[0])
    
    '''

    # preprocessing
    xshape = x.get_shape().as_list()
    noise = tf.get_variable('noise', xshape, tf.float32,
                            initializer=tf.initializers.zeros) # optimization variables
    x_scaled = (x - clip[0]) / (clip[1] - clip[0])
    z = tf.clip_by_value(x_scaled, 1e-8, 1-1e-8)
    xinv = tf.log(z / (1 - z)) / T 

    # update samples
    xadv = tf.sigmoid(T * (xinv + noise)) 
    xadv = xadv * (clip[1] - clip[0]) + clip[0] 

    ybar, logits = model(xadv, logits=True) # Model is only used for evaluate the logit
    ydim = ybar.get_shape().as_list()[1]

    # decide target labels
    if y is not None:
        y = tf.cond(tf.equal(tf.rank(y), 0),
                    lambda: tf.fill([xshape[0]], y),
                    lambda: tf.identity(y))
    else:
        y = tf.argmin(ybar, axis=1, output_type=tf.int32)

    # evaluate attack
    mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf')) # why target label has probability 0
    yt = tf.reduce_max(logits - mask, axis=1) # why care
    yo = tf.reduce_max(logits, axis=1) # why care
    loss0 = tf.nn.relu(yo - yt + min_prob) # if yo-yt < -min_prob or yt-yo > min_prob, the attack is good enough, no penalty loss
    axis = list(range(1, len(xshape)))
    print(f'mask {mask}')
    print(f'yt {yt}')
    print(f'yo {yo}')
    print(f'axis {axis}')

    # distance check
    loss1 = tf.reduce_mean(tf.square(xadv-x)) # l2 distance loss
    loss = eps*loss0 + loss1
    train_op = optimizer.minimize(loss, var_list=[noise]) # you can specify your own loss and optimizationvariables


    return train_op, xadv, noise

