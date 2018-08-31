import tensorflow as tf


def weigthed_l1_loss(left, right, y):
    dist = tf.abs(left - right)

    x = tf.layers.dense(dist, 1, use_bias=False)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=x)

    return dist, tf.reduce_mean(loss)


def contrastive_loss(left, right, y, margin):
    dist_sqr = tf.reduce_sum(tf.pow(left - right, 2))
    dist = tf.sqrt(dist_sqr)

    loss = y * dist_sqr + (1 - y) * tf.maximum(0., tf.pow(margin - dist, 2))

    return dist, tf.reduce_mean(loss)


def triplet_ranking_loss(positive, middle, negative, margin):
    def dist(a, b):
        return tf.sqrt(tf.reduce_sum(tf.pow(a - b, 2)))

    dist_pm = dist(positive, middle)
    dist_pn = dist(positive, negative)
    dist_mn = dist(middle, negative)

    loss = tf.maximum(0, dist_pm + margin - dist_mn) +\
           tf.maximum(0, dist_pm + margin - dist_pn)

    return (dist_pm, dist_mn), tf.reduce_mean(loss)