from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
import load_data as data
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './data/logs/', """Directory where to write event logs. """)
tf.app.flags.DEFINE_integer('max_steps', 10000, """Number of epochs to run.""")
tf.app.flags.DEFINE_string('save_dir', './data/saves/', """Directory where to save the checkpoints. """)
tf.app.flags.DEFINE_string('tfrecord_file', './data/tfrecord_file', """File with the dataset to train. """)

num_classes = 2
dropout_keep_prob = 0.95
learning_rate = 0.0005


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def train():
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        iterator = data.create_iterator_for_diff(FLAGS.tfrecord_file)
        print(iterator)

        bottlenecks_1_batch, bottlenecks_2_batch, labels_batch = iterator.get_next()
        print("bottleneck_1:", bottlenecks_1_batch)
        print("bottleneck_2:", bottlenecks_2_batch)
        print("labels_batch:", labels_batch)

        diff_bottlenecks_tensor = tf.subtract(bottlenecks_1_batch, bottlenecks_2_batch)
        print("Difference: ", diff_bottlenecks_tensor)

        with tf.variable_scope('classify'):
            end_points = {}

            net = slim.dropout(diff_bottlenecks_tensor, dropout_keep_prob, scope='Dropout_1b')
            net = slim.flatten(net, scope='Flatten_1')

            end_points['Flatten_1'] = net

            # Creates a fully connected layer
            net = slim.fully_connected(net, 512, activation_fn=None,
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       scope='FC_1')

            tf.summary.histogram(name='Weights',
                                 values=tf.get_default_graph().get_tensor_by_name('classify/FC_1/weights:0'))
            tf.summary.histogram(name='Biases',
                                 values=tf.get_default_graph().get_tensor_by_name('classify/FC_1/biases:0'))

            net = slim.dropout(net, dropout_keep_prob, scope='Dropout_2b')
            net = slim.flatten(net, scope='Flatten_2')

            end_points['Flatten_2'] = net

            net = slim.fully_connected(net, num_classes, activation_fn=None,
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       scope='FC_2')

            print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))

            tf.summary.histogram(name='Weights',
                                 values=tf.get_default_graph().get_tensor_by_name('classify/FC_2/weights:0'))
            tf.summary.histogram(name='Biases',
                                 values=tf.get_default_graph().get_tensor_by_name('classify/FC_2/biases:0'))

            end_points['Logits'] = net

            tf.add_to_collection('classify', net)

        # Loss calculation
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels_batch)
        cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy')
        tf.summary.scalar(name='loss', tensor=cross_entropy_mean)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cross_entropy_mean, global_step=global_step,
                                      var_list=tf.global_variables('classify'))

        # Savers
        saver_ft = tf.train.Saver(tf.global_variables('classify'))

        # Initializer
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Tensorborad options
            train_writer = tf.summary.FileWriter(FLAGS.log_dir, g)

            logger = init_logger()
            logger.info("Training starts...")

            # Training loop. Set the max number of steps.
            for epoch in range(0, FLAGS.max_steps):
                # We compute the image and label batch
                sess.run([diff_bottlenecks_tensor, labels_batch])

                # Merge all summary variables for Tensorborad
                merge = tf.summary.merge_all()

                # We do the training and compute the loss and the summaries
                _, loss_val, summary = sess.run([train_op, cross_entropy_mean, merge])

                if epoch % 10 is 0:
                    logger.info('Time: %s   Loss: %f   Step: %i', str(datetime.now()), loss_val, epoch)
                    # Write the summaries in the log file
                    train_writer.add_summary(summary, epoch)

                # We save the progress every 500 steps
                if epoch % 500 is 0 and epoch is not 0:
                    saver_ft.save(sess, FLAGS.save_dir, global_step=global_step)
                    logger.info("***** Saving model in: %s *****", FLAGS.save_dir)

            logger.info("Training ends...")
            saver_ft.save(sess, FLAGS.save_dir, global_step=global_step)
            logger.info("***** Saving model in: %s *****", FLAGS.save_dir)


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
