# TensorFlow implementation of the Guided Anisotropic Diffusion algorithm
# Original PyTorch implementation by Rodrigo Caye Daudt (https://rcdaudt.github.io)
# Reference paper:
# Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau.
# "Guided anisotropic diffusion and iterative learning for weakly supervised change detection."
# In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019.

import tensorflow as tf


def g(x, K=5):
    return 1.0 / (1.0 + (tf.abs((x * x)) / (K * K)))


def get_image_gradients(image):
    dv = image[:, :, 1:, 1:-1] - image[:, :, :-1, 1:-1]
    dh = image[:, :, 1:-1, 1:] - image[:, :, 1:-1, :-1]
    return dv, dh


def diffusion_coefficient(gradient_v, gradient_h, K):
    cv = g(tf.reduce_mean(gradient_v, 1), K)
    ch = g(tf.reduce_mean(gradient_h, 1), K)
    return cv, ch


@tf.function
def diffuse(image, lambda_, K, coeffs=None, return_coeffs=False):
    # Compute gradients from the image
    dv, dh = get_image_gradients(image)
    if coeffs is None:
        cv, ch = diffusion_coefficient(dv, dh, K)
    else:
        cv, ch = coeffs
    diffused = image[:, :, 1:-1, 1:-1] + lambda_ * (
        cv[:, 1:, :] * dv[:, :, 1:, :]
        - cv[:, :-1, :] * dv[:, :, :-1, :]
        + ch[:, :, 1:] * dh[:, :, :, 1:]
        - ch[:, :, :-1] * dh[:, :, :, :-1]
    )
    image = tf.pad(diffused, tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]]))
    if return_coeffs:
        return image, cv, ch
    else:
        return image


@tf.function
def anisotropic_diffusion(
    input_image,
    first_guide,
    second_guide=None,
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
        first_guide, cv1, ch1 = diffuse(first_guide, lambda_, K, return_coeffs=True)

        # Perform diffusion on the second guide (if specified)
        if second_guide is not None:
            second_guide, cv2, ch2 = diffuse(second_guide, lambda_, K, return_coeffs=True)
            cv = tf.math.minimum(cv1, cv2)
            ch = tf.math.minimum(ch1, ch2)
        else:
            cv, ch = cv1, ch1

        input_image = diffuse(input_image, lambda_, K, coeffs=(cv, ch))

    return input_image
