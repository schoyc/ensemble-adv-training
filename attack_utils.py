import numpy as np
import keras.backend as K

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

import tensorflow as tf


def linf_loss(X1, X2):
    return np.max(np.abs(X1 - X2), axis=(1, 2, 3))


def gen_adv_loss(logits, y, loss='logloss', mean=False):
    """
    Generate the loss function.
    """

    if loss == 'training':
        # use the model's output instead of the true labels to avoid
        # label leaking at training time
        y = tf.cast(tf.equal(logits, tf.max(logits, 1, keepdims=True)), tf.float32)
        y = y / tf.sum(y, 1, keepdims=True)
        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    elif loss == 'logloss':
        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    else:
        raise ValueError("Unknown loss: {}".format(loss))

    if mean:
        out = tf.mean(out)
    else:
        out = tf.sum(out)
    return out


def gen_grad(x, logits, y, loss='logloss'):
    """
    Generate the gradient of the loss function.
    """

    adv_loss = gen_adv_loss(logits, y, loss)

    # Define gradient of loss wrt input
    grad = tf.gradients(adv_loss, [x])[0]
    return grad
