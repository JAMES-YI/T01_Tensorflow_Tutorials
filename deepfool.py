import tensorflow as tf
'''
From: https://github.com/gongzhitaao/tensorflow-adversarial
Modified by JYI
(1) add comments
(2) remove generation of adversarial samples in batch mode
'''


def deepfool(model, x, noise=False, eta=0.01, epochs=3,
             clip_min=0.0, clip_max=1.0, min_prob=0.0):
    """DeepFool implementation in Tensorflow.

    The original DeepFool will stop whenever we successfully cross the
    decision boundary.  Thus it might not run total epochs.  In order to force
    DeepFool to run full epochs, you could set batch=True.  In that case the
    DeepFool will run until the max epochs is reached regardless whether we
    cross the boundary or not.  See https://arxiv.org/abs/1511.04599 for
    details.

    :param model: Model function.
    :param x: 2D or 4D input tensor.
    :param noise: Also return the noise if True.
    :param eta: Small overshoot value to cross the boundary.
    :param epochs: Maximum epochs to run.
    :param batch: If True, run in batch mode, will always run epochs.
    :param clip_min: Min clip value for output.
    :param clip_max: Max clip value for output.
    :param min_prob: Minimum probability for adversarial samples.

    :return: Adversarials, of the same shape as x.
    """

    '''JYI
    1 how to transfer a model function
    2 how the search for adversarial examples is performed
    3 how to perform targeted attack and untargeted attck
    '''
    y = tf.stop_gradient(model(x))
    fn = _deepfoolx # <function _deepfoolx at 0x7ff105610730>

    def _f(xi):
        xi = tf.expand_dims(xi, axis=0)
        z = fn(model, xi, eta=eta, epochs=epochs, clip_min=clip_min,
               clip_max=clip_max, min_prob=min_prob)
        return z[0]

    delta = tf.map_fn(_f, x, dtype=(tf.float32), back_prop=False,
                      name='deepfool')

    if noise:
        return delta

    xadv = tf.stop_gradient(x + delta*(1+eta))
    xadv = tf.clip_by_value(xadv, clip_min, clip_max)
    return xadv


def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret


def _deepfoolx(model, x, epochs, eta, clip_min, clip_max, min_prob):
    """DeepFool for multi-class classifiers.
    Assumes that the final label is the label with the maximum values.
    """
    '''JYI
    1 how to transfer model
    2 why epochs exists here
    3 motivation for min_prob
    4 how the input arguments of _cond are specified
    '''

    y0 = tf.stop_gradient(model(x))
    y0 = tf.reshape(y0, [-1])
    k0 = tf.argmax(y0)

    ydim = y0.get_shape().as_list()[0] # 10 corresponding to 10 classes
    xdim = x.get_shape().as_list()[1:]
    xflat = _prod(xdim)

    def _cond(i, z):
        '''JYI
        _cond is false and terminate the loop if
        (1) i>=epochs
        or
        (2) the following two hold
        (2.1) k0 is not equal to k
        (2.2) p is greater than min_prob

        it may fail to generate adversarial samples

        nested function can use the variables without explicitly transferring

        '''
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])
        p = tf.reduce_max(y)
        k = tf.argmax(y)
        return tf.logical_and(tf.less(i, epochs),
                              tf.logical_or(tf.equal(k0, k),
                                            tf.less(p, min_prob)))

    def _body(i, z):
        '''JYI
        1 the output of body updates the loop_variables
        2 perturbation path: 0, dx,

        gs [<tf.Tensor 'model_1/deepfool/while/_deepfoolx/Reshape_11:0' shape=(784,) dtype=float32>]

        '''
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])

        gs = [tf.reshape(tf.gradients(y[i], xadv)[0], [-1]) for i in range(ydim)] # Jacobian probability wrt input image
        g = tf.stack(gs, axis=0)

        yk, yo = y[k0], tf.concat((y[:k0], y[(k0+1):]), axis=0)
        gk, go = g[k0], tf.concat((g[:k0], g[(k0+1):]), axis=0)

        yo.set_shape(ydim - 1)
        go.set_shape([ydim - 1, xflat])

        a = tf.abs(yo - yk) # difference between the probability of adversarial label and other labels
        b = go - gk
        print(f'b {b}')
        c = tf.norm(b, axis=1)
        score = a / c
        ind = tf.argmin(score)

        si, bi = score[ind], b[ind]
        dx = si * bi
        dx = tf.reshape(dx, [-1] + xdim)
        return i+1, z+dx

    _, noise = tf.while_loop(cond=_cond, body=_body, loop_vars=[0, tf.zeros_like(x)],
                             name='_deepfoolx', back_prop=False)
    return noise
