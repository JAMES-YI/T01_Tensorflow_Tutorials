import tensorflow as tf

'''
from: https://github.com/gongzhitaao/tensorflow-adversarial
Modified by JYI, 06/10/2019
- (1) add comments
- (2) comment out function for changing two pixels at each iteration
'''

__all__ = ['jsma']


def jsma(model, x, y=None, epochs=1, eps=1.0, clip_min=0.0, clip_max=1.0,
         score_fn=lambda t, o: t * tf.abs(o)):
    """
    Jacobian-based saliency map approach.

    See https://arxiv.org/abs/1511.07528 for details.  During each iteration,
    this method finds the pixel (or two pixels) that has the most influence on
    the result (most salient pixel) and add noise to the pixel.

    :param model: A wrapper that returns the output tensor of the model.
    :param x: The input placeholder a 2D or 4D tensor.
    :param y: The desired class label for each input, either an integer or a
              list of integers (each corresponds a sample).
    :param epochs: Maximum epochs to run.  If it is a floating number in [0,
        1], it is treated as the distortion factor, i.e., gamma in the
        original paper.
    :param eps: The noise added to input per epoch.
    :param clip_min: The minimum value in output tensor.
    :param clip_max: The maximum value in output tensor.
    :param score_fn: Function to calculate the saliency score.

    :return: A tensor, contains adversarial samples for each input.
    """
    '''JYI
    how the score function is used
    how the model is transferred
    how the function is transferred? just transfer the function name
    y Tensor("model_1/adv_y:0", shape=(), dtype=int32)
    purpose of target

    '''
    n = tf.shape(x)[0] # Tensor("model_1/strided_slice:0", shape=(), dtype=int32)

    target = tf.cond(pred=tf.equal(0, tf.rank(y)),
                     true_fn=lambda: tf.zeros([n], dtype=tf.int32) + y,
                     false_fn=lambda: y) # the value of pred determines what to return
    target = tf.stack((tf.range(n), target), axis=1) # target Tensor("model_1/stack:0", shape=(?, 2), dtype=int32)

    if isinstance(epochs, float):
        tmp = tf.to_float(tf.size(x[0])) * epochs
        epochs = tf.to_int32(tf.floor(tmp))

    _jsma_fn = _jsma_impl

    return _jsma_fn(model, x, target, epochs=epochs, eps=eps,
                    clip_min=clip_min, clip_max=clip_max, score_fn=score_fn)


def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret


def _jsma_impl(model, x, yind, epochs, eps, clip_min, clip_max, score_fn):
    '''JYI
    every time change one pixel
    dy_dx Tensor("model_1/_jsma_batch/gradients/model_1/_jsma_batch/conv0/conv2d/Conv2D_grad/Conv2DBackpropInput:0",
                 shape=(?, 28, 28, 1), dtype=float32)
    dt_dx Tensor("model_1/_jsma_batch/gradients_1/model_1/_jsma_batch/conv0/conv2d/Conv2D_grad/Conv2DBackpropInput:0",
                 shape=(?, 28, 28, 1), dtype=float32)
    tf.gradients(yt,xadv) list, [<tf.Tensor 'model_1/_jsma_batch/gradients_2/model_1/_jsma_batch/conv0/conv2d/Conv2D_grad/Conv2DBackpropInput:0'
                           shape=(?, 28, 28, 1) dtype=float32>]

    difference betwen yind, ybar, yt
    c0
    c1
    '''

    def _cond(i, xadv): # even though not used, you should list the loop variables; terminate when i>epochs
        return tf.less(i, epochs)

    def _body(i, xadv):
        ybar = model(xadv)

        dy_dx = tf.gradients(ybar, xadv)[0] # specify the calculation of gradient; logit to input;

        # gradient separation
        yt = tf.gather_nd(params=ybar, indices=yind) # only take the index corresponding to target class
        dt_dx = tf.gradients(yt, xadv)[0] # gradients of target w.r.t input
        do_dx = dy_dx - dt_dx # gradients of non-targets w.r.t input

        c0 = tf.logical_or(eps < 0, xadv < clip_max)
        c1 = tf.logical_or(eps > 0, xadv > clip_min)
        cond = tf.reduce_all([dt_dx >= 0, do_dx <= 0, c0, c1], axis=0)
        cond = tf.to_float(cond)

        # saliency score for each pixel
        score = cond * score_fn(dt_dx, do_dx)

        shape = score.get_shape().as_list()
        dim = _prod(shape[1:])
        score = tf.reshape(score, [-1, dim])

        # find the pixel with the highest saliency score
        ind = tf.argmax(score, axis=1)
        dx = tf.one_hot(ind, dim, on_value=eps, off_value=0.0)
        dx = tf.reshape(dx, [-1] + shape[1:])

        xadv = tf.stop_gradient(xadv + dx)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)

        return i+1, xadv

    _, xadv = tf.while_loop(_cond, _body, (0, tf.identity(x)),
                            back_prop=False, name='_jsma_batch')

    return xadv
