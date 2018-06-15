from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging
from prueba1 import data_loader as data
from model import inception_resnet_v1 as model
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_dir', '/home/uc3m1/PycharmProjects/siameseFaceNet/weights/model-20180408-102900.ckpt-90', """Directory where to restore a model""")
tf.app.flags.DEFINE_string('log_dir', '/home/uc3m1/PycharmProjects/siameseFaceNet/prueba1/logs/', """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('max_steps', 500, """Number of epochs to run.""")
tf.app.flags.DEFINE_string('save_dir', '/home/uc3m1/PycharmProjects/siameseFaceNet/prueba1/save/', """Directory where to save the checkpoints """)


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def train():
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        paths, labels = data.txt_to_np()

        print("Paths:", paths)
        print("Labels: ", labels)
        iterator = data.create_iterator(paths, labels)
        print(iterator)
        images_batch, labels_batch = iterator.get_next()
        print("images_batch:", images_batch)
        print("labels_batch:", labels_batch)

        # Making the bottlenecks
        bottleneck, end_points = model.inference(images_batch)

        # Obtaining the logits of the fully connected layer with the bottlenecks
        logits = model.fine_tuning(bottleneck, end_points, num_classes= 7)
        # TODO: automatizr el numero de clases en funcion de las que tengamosen cada caso

        print("Bottleneck: ", bottleneck)
        print("Logits: ", logits)
        print("Labels: ", labels_batch)

        # Loss calculation
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_batch)
        cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy')
        tf.summary.scalar(name='loss', tensor=cross_entropy_mean)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(0.005)
        train_op = optimizer.minimize(cross_entropy_mean, global_step=global_step, var_list=tf.global_variables('fine_tuning'))

        # Savers
        saver = tf.train.Saver(tf.global_variables("InceptionResnetV1"))
        # todo: automatizar nombre del modelo
        saver_ft = tf.train.Saver(tf.global_variables('fine_tuning'))

        # Initializer
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            print(FLAGS.ckpt_dir)
            print(tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            saver.restore(sess, FLAGS.ckpt_dir)

            # This will let you see the images in tensorboard
            tf.summary.image(tensor=images_batch, name="Image")

            # Tensorborad options
            train_writer = tf.summary.FileWriter(FLAGS.log_dir, g)

            logger = init_logger()
            logger.info("Training starts...")

            # Training loop. Set the max number of steps.
            for epoch in range(0, FLAGS.max_steps):
                # We compute the image and label batch
                sess.run([images_batch, labels_batch])

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
