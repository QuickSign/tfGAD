# Guided Anisotropic Diffusion algorithm
# Rodrigo Caye Daudt
# https://rcdaudt.github.io
#
# Caye Daudt, Rodrigo, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau. "Guided anisotropic diffusion and iterative learning for weakly supervised change detection." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 0-0. 2019.

import tensorflow as tf


def g(x, K=5):
    return 1.0 / (1.0 + (tf.abs((x * x)) / (K * K)))


def get_gradients(image):
    dv = image[:, :, 1:, 1:-1] - image[:, :, :-1, 1:-1]
    dh = image[:, :, 1:-1, 1:] - image[:, :, 1:-1, :-1]
    # dv = image[:, :, 1:, :] - image[:, :, :-1, :]
    # dh = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dv, dh


def diffusion_coefficient(image, K=5):
    dv, dh = get_gradients(image)
    cv = g(tf.reduce_mean(dv, 1), K)
    ch = g(tf.reduce_mean(dh, 1), K)
    return cv, ch


def diffuse(cv, ch, image, lambda_):
    channels = []
    # Compute gradients from the image
    dv, dh = get_gradients(image)
    for channel in range(int(image.shape[1])):
        diffused_channel = image[:, channel, 1:-1, 1:-1] + lambda_ * (
            cv[:, 1:, :] * dv[:, channel, 1:, :]
            - cv[:, :-1, :] * dv[:, channel, :-1, :]
            + ch[:, :, 1:] * dh[:, channel, :, 1:]
            - ch[:, :, :-1] * dh[:, channel, :, :-1]
        )
        channels.append(diffused_channel)
    del (dv, dh)
    return tf.pad(tf.stack(channels, axis=1), tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]]))


def anisotropic_diffusion(
    input_image,
    first_guide_image,
    second_guide_image,
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

        channels = []

        # Compute gradients from the guide image
        dv = first_guide_image[:, :, 1:, 1:-1] - first_guide_image[:, :, :-1, 1:-1]
        dh = first_guide_image[:, :, 1:-1, 1:] - first_guide_image[:, :, 1:-1, :-1]
        for channel in range(int(first_guide_image.shape[1])):
            diffused_channel = first_guide_image[:, channel, 1:-1, 1:-1] + lambda_ * (
                cv1[:, 1:, :] * dv[:, channel, 1:, :]
                - cv1[:, :-1, :] * dv[:, channel, :-1, :]
                + ch1[:, :, 1:] * dh[:, channel, :, 1:]
                - ch1[:, :, :-1] * dh[:, channel, :, :-1]
            )
            channels.append(diffused_channel)
        first_guide_image = tf.pad(
            tf.stack(channels, axis=1), tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])
        )
        del (dv, dh)

        cv2, ch2 = diffusion_coefficient(second_guide_image, K=K)

        channels = []
        dv = second_guide_image[:, :, 1:, 1:-1] - second_guide_image[:, :, :-1, 1:-1]
        dh = second_guide_image[:, :, 1:-1, 1:] - second_guide_image[:, :, 1:-1, :-1]
        for channel in range(int(second_guide_image.shape[1])):
            diffused_channel = second_guide_image[:, channel, 1:-1, 1:-1] + lambda_ * (
                cv2[:, 1:, :] * dv[:, channel, 1:, :]
                - cv2[:, :-1, :] * dv[:, channel, :-1, :]
                + ch2[:, :, 1:] * dh[:, channel, :, 1:]
                - ch2[:, :, :-1] * dh[:, channel, :, :-1]
            )
            channels.append(diffused_channel)
        second_guide_image = tf.pad(
            tf.stack(channels, axis=1), tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])
        )
        del (dv, dh)

        cv = tf.math.minimum(cv1, cv2)
        ch = tf.math.minimum(ch1, ch2)
        del (cv1, ch1, cv2, ch2)

        dv = input_image[:, :, 1:, 1:-1] - input_image[:, :, :-1, 1:-1]
        dh = input_image[:, :, 1:-1, 1:] - input_image[:, :, 1:-1, :-1]
        channels = []
        for channel in range(int(input_image.shape[1])):
            diffused_channel = input_image[:, channel, 1:-1, 1:-1] + lambda_ * (
                cv[:, 1:, :] * dv[:, channel, 1:, :]
                - cv[:, :-1, :] * dv[:, channel, :-1, :]
                + ch[:, :, 1:] * dh[:, channel, :, 1:]
                - ch[:, :, :-1] * dh[:, channel, :, :-1]
            )
            channels.append(diffused_channel)
        input_image = tf.pad(
            tf.stack(channels, axis=1), tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])
        )
        del (dv, dh)

    return input_image
