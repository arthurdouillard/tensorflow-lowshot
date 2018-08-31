#!/usr/bin/env python3
import argparse

import tensorflow as tf
import tqdm

import lowshot


def parse_args():
    parser = argparse.ArgumentParser(description='Train a lowshot model.')
    parser.add_argument('--iterations', action='store', type=int, default=10000,
                        help='Number of iterations.')
    parser.add_argument('--step', action='store', type=int, default=500,
                        help='Print logs every X iterations.')
    parser.add_argument('--size', action='store', type=int, default=105,
                        help='Target size for a square image.')
    parser.add_argument('--batch_size', action='store', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--save', action='store', default=None,
                        help='Save path for the model.')

    return parser.parse_args()


def train(args):
    train_set, test_set, _ = lowshot.loaders.parse_omniglot('data/omniglot')
    loader = lowshot.loaders.SiameseLoader(*train_set, batch_size=args.batch_size,
                                           target_size=(args.size, args.size))


    filters = [[16, 16], [32, 32, 32], [64, 64, 64]]

    left = tf.placeholder(tf.float32, shape=[None, args.size, args.size, 3])
    right = tf.placeholder(tf.float32, shape=[None, args.size, args.size, 3])
    similarity = tf.placeholder(tf.float32, shape=[None])

    left_embedding = lowshot.embeddings.convnet(left, filters)
    right_embedding = lowshot.embeddings.convnet(right, filters)

    dist, closs = lowshot.loss.contrastive_loss(left_embedding, right_embedding, similarity, 1.0)

    opt = tf.train.AdamOptimizer().minimize(closs)

    tf.add_to_collection('left', left)
    tf.add_to_collection('right', right)
    tf.add_to_collection('dist', dist)

    if args.save:
        saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'convnet*')
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_loss = float('inf')
        for epoch in range(args.iterations // args.step):
            epoch_loss = 0.

            progress_bar = tqdm.trange(args.step, desc='Metrics')
            for i in progress_bar:
                actual_left, actual_right, actual_simi = loader.get_batch()

                _, loss = sess.run([opt, closs], feed_dict={
                    left: actual_left,
                    right: actual_right,
                    similarity: actual_simi
                })

                epoch_loss += loss

                progress_bar.set_description('Loss = {}'.format(
                    round(epoch_loss / (i+1), 4)
                ))

            epoch_loss /= args.step
            if epoch_loss < best_loss:
                print(f'Improving loss {best_loss} --> {epoch_loss}, saving model...', end=' ')
                saver.save(sess, args.save)
                print('Done!')



if __name__ == '__main__':
    args = parse_args()
    train(args)