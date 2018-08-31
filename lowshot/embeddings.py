import tensorflow as tf

def convnet(input, filters, bn=True, dropout=False, reuse=tf.AUTO_REUSE,
            extract='flatten', normalize='l2'):
    """Creates a convnet with standard convolutions.

    # Arguments:
        input: The input tensor of shape [None, w, h, c].
        n_filters: A list of list of integers.
                   Each sublist represents a block between pooling.
                   Each integer represents the number of filters.
        bn: Boolean to activate batch normalization.
        dropout: Amount of dropout to use.
        reuse: Variable reuse, useful for siamese, triplet, etc.
        extract: Method to extract the spatial features to a 1d vector:
                    - 'flatten', 'max', 'avg', callable.
        normalize: Normalize the 1d features vector:
                    - 'l2', False, callable.
    """
    x = input

    with tf.variable_scope('convnet'):
        for block_id, block_filters in enumerate(filters):
            with tf.variable_scope(f'block{str(block_id)}'):
                x = block(x, block_filters, bn=bn, dropout=dropout, reuse=reuse)

    if extract == 'flatten':
        x = tf.layers.flatten(x)
    elif extract == 'max':
        x = tf.reduce_max(x, [1, 2])
    elif extract == 'avg':
        x = tf.reduce_mean(x, [1, 2])
    elif callable(extract):
        x = extract(x)
    else:
        raise ValueError(f'Extract function {str(extract)} is invalid.')

    if normalize == 'l2':
        x = tf.nn.l2_normalize(x)
    elif callable(normalize):
        x = normalize(x)
    elif not normalize:
        pass
    else:
        raise ValueError(f'Normalize function {str(extract)} is invalid.')

    return x


def block(x, filters, bn, dropout, reuse):
    for conv_id, f in enumerate(filters):
        with tf.variable_scope(f'conv{str(conv_id)}'):
            x = tf.layers.conv2d(x, f, 3, reuse=reuse)

            if bn:
                x = tf.layers.batch_normalization(x, reuse=reuse)

            x = tf.nn.relu(x)

            if dropout:
                x = tf.layers.dropout(x, dropout, reuse=reuse)

    return x