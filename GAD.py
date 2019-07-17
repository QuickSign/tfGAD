# TensorFlow implementation of the Guided Anisotropic Diffusion algorithm
# Original PyTorch implementation by Rodrigo Caye Daudt (https://rcdaudt.github.io)
# Reference paper:
# Caye Daudt, Rodrigo, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau.
# "Guided anisotropic diffusion and iterative learning for weakly supervised change detection."
# In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019.

import tensorflow as tf


def g(x, K=5):
    return 1.0 / (1.0 + (tf.abs((x * x)) / (K * K)))


def get_gradients(image):
    dv = image[:, :, 1:, 1:-1] - image[:, :, :-1, 1:-1]
    dh = image[:, :, 1:-1, 1:] - image[:, :, 1:-1, :-1]
    return dv, dh


def diffusion_coefficient(image, K=5):
    dv, dh = get_gradients(image)
    cv = g(tf.reduce_mean(dv, 1), K)
    ch = g(tf.reduce_mean(dh, 1), K)
    return cv, ch


def diffuse(cv, ch, image, lambda_):
    # Compute gradients from the image
    dv, dh = get_gradients(image)
    diffused = image[:, :, 1:-1, 1:-1] + lambda_ * (
        cv[:, 1:, :] * dv[:, :, 1:, :]
        - cv[:, :-1, :] * dv[:, :, :-1, :]
        + ch[:, :, 1:] * dh[:, :, :, 1:]
        - ch[:, :, :-1] * dh[:, :, :, :-1]
    )
    del (dv, dh)
    return tf.pad(diffused, tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]]))


def anisotropic_diffusion(
    input_image,
    first_guide_image,
    second_guide_image=None,
    iterations=500,
    lambda_=0.24,
    K=5,
    is_log=True,
    verbose=False,
):

    if is_log:
        input_image = tf.math.exp(input_image)

    for t in range(iterations):
        if verbose:
            print("Iteration {}".format(t))

        # Perform diffusion on the first guide
        cv1, ch1 = diffusion_coefficient(first_guide_image, K=K)
        first_guide_image = diffuse(cv1, ch1, first_guide_image, lambda_)

        # Perform diffusion on the second guide (if specified)
        if second_guide_image is not None:
            cv2, ch2 = diffusion_coefficient(second_guide_image, K=K)
            second_guide_image = diffuse(cv2, ch2, second_guide_image, lambda_)

            cv = tf.math.minimum(cv1, cv2)
            ch = tf.math.minimum(ch1, ch2)
            del (cv1, ch1, cv2, ch2)
        else:
            cv, ch = cv1, ch1

        input_image = diffuse(cv, ch, input_image, lambda_)

    return input_image
