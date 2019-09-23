import tensorflow as tf

from misc.registry import register_model
from classification.misc.layers import conv2d


@register_model("unet")
def unet(inputs, controller, is_training, batch_norm, layer_collection, particles):
    def unet_layer(inputs, filter_size, out_channel):
        l2_loss = 0.
        in_channel = inputs.shape.as_list()[-1]
        container, idx = controller.register_conv_layer(filter_size, in_channel, out_channel)
        sampled_weight = controller.get_weight(idx)
        pre, act = conv2d(inputs, sampled_weight, (filter_size, filter_size, in_channel, out_channel),
                          batch_norm, is_training, particles, padding="SAME")
        layer_collection.register_conv2d(controller.get_params(idx), (1, 1, 1, 1), "SAME", inputs, pre)
        l2_loss += 0.5 * tf.reduce_sum(sampled_weight ** 2)
        return pre, act, l2_loss

    x = tf.tile(inputs, [particles, 1, 1, 1])

    # if batch_norm_input:
    # x = tf.layers.batch_normalization(x, training=is_training, momentum=0.999, epsilon=1e-5)

    net = {}
    num_classes = 4
    num_levels = 4
    num_blocks = [2, 2, 2, 2]
    num_filters = [32, 64, 128, 256, 16]
    filter_size = 3

    total_l2_loss = 0.0

    # Contracting path
    for level in range(num_levels):
        for block in range(num_blocks[level]):

            print('Adding conv{}_{}'.format(level, block))

            _, act, l2_loss = unet_layer(x, filter_size, num_filters[level])
            x = act
            total_l2_loss += l2_loss

        net['conv{}_{}'.format(level, block)] = x

        if not level == (num_levels - 1):
            print('Adding maxpool')
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(inputs=x)

    # Expanding path
    for level in range(num_levels - 2, -1, -1):
        print('Adding upsampling')
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(inputs=x)

        print('Concatenating with conv{}_{}'.format(level, num_blocks[level] - 1))
        x = tf.concat([net['conv{}_{}'.format(level, num_blocks[level] - 1)], x], axis=-1)

        for block in range(num_blocks[level]):
            print('Adding upconv{}_{}'.format(level, block))
            _, act, l2_loss = unet_layer(x, filter_size, num_filters[level])
            x = act
            total_l2_loss += l2_loss

        net['upconv{}_{}'.format(level, block)] = x

    print('Adding 2 final 1x1 conv')

    _, act, l2_loss = unet_layer(x, filter_size=1, out_channel=num_filters[level])
    x = act
    total_l2_loss += l2_loss

    pre, act, l2_loss = unet_layer(x, filter_size=1, out_channel=num_classes)
    total_l2_loss += l2_loss

    layer_collection.register_categorical_predictive_distribution(pre, name="logits")

    return pre, total_l2_loss
