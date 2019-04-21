
import keras.backend as K
from attack_utils import gen_grad
import tensorflow as tf


def symbolic_fgs(x, grad, eps=0.3, clipping=True):
    """
    FGSM attack.
    """

    # signed gradient
    normed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x


def iter_fgs(model, x, y, steps, eps):
    """
    I-FGSM attack.
    """

    adv_x = x

    # iteratively apply the FGSM with small step size
    for i in range(steps):
        logits = model(adv_x)
        grad = gen_grad(adv_x, logits, y)

        adv_x = symbolic_fgs(adv_x, grad, eps, True)
    return adv_x
