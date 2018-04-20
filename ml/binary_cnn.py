# pylint: disable=E0632

import tensorflow as tf
import numpy as np
from .embeddings import seq_from_matrix
import random
from .dao import HDF5Dao, HDF5TargetDao
from sys import argv, exit


tf.logging.set_verbosity(tf.logging.WARN)

SEQ_LEN = 2500
NUM_CHARS = 20
NUM_TARGETS = 2

if __name__ == "__main__":
    try:
        train_path, function, taxon_id = argv[1:]
        assert function in {"motility", "biofilm"}
        assert taxon_id in {"208963", "237561"}
    except (ValueError, AssertionError):
        print("Usage: python -m ml.binary_cnn "
            "path/to/train/data.h5 [motility|biofilm] [208963|237561]")
        exit()

    random.seed(0)
    inputs = tf.placeholder(tf.float32, [None, SEQ_LEN, NUM_CHARS], "sequence")
    targets = tf.placeholder(tf.float32, [None, NUM_TARGETS])
    conv0 = tf.layers.conv1d(inputs=inputs, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv1 = tf.layers.conv1d(inputs=conv0, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv1 = tf.layers.batch_normalization(conv1)
    conv2 = tf.layers.conv1d(inputs=conv1, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv2 = tf.add(conv0, conv2)
    conv3 = tf.layers.conv1d(inputs=conv2, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.batch_normalization(conv3)
    conv4 = tf.layers.conv1d(inputs=conv3, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv4 = tf.add(conv2, conv4)
    conv5 = tf.layers.conv1d(inputs=conv4, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv5 = tf.layers.batch_normalization(conv5)
    conv6 = tf.layers.conv1d(inputs=conv5, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv6 = tf.add(conv4, conv6)
    conv7 = tf.layers.conv1d(inputs=conv6, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv7 = tf.layers.batch_normalization(conv7)
    conv8 = tf.layers.conv1d(inputs=conv7, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv8 = tf.add(conv6, conv8)
    conv9 = tf.layers.conv1d(inputs=conv8, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv9 = tf.layers.batch_normalization(conv9)
    conv10 = tf.layers.conv1d(inputs=conv9, filters=64, kernel_size=(5,), padding="same", activation=tf.nn.relu)
    conv10 = tf.add(conv8, conv10)
    flattened = tf.contrib.layers.flatten(conv10)
    fc1 = tf.contrib.layers.fully_connected(flattened, 128)
    outputs = tf.contrib.layers.fully_connected(fc1, NUM_TARGETS, activation_fn=None)
    vars_ = tf.trainable_variables()
    l2 = tf.add_n([
        tf.nn.l2_loss(v) for v in vars_
            if 'bias' not in v.name
    ]) * 0.001
    loss = tf.losses.softmax_cross_entropy(targets, outputs) + l2
    tf.summary.scalar("mse", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("log/binary/residual/train", sess.graph)
        test_writer = tf.summary.FileWriter("log/binary/residual/test", sess.graph)
        tf.global_variables_initializer().run()
        #dao = HDF5Dao("./data/parsed/all_train.h5", label_type="binary/biofilm")
        dao = HDF5Dao(train_path, label_type="binary/{}".format(function))
        target_dao = HDF5TargetDao("./data/parsed/target.{}.h5".format(taxon_id))
        batch_size = 100

        i = 0
        num_epochs = 10.0
        while dao.epochs < num_epochs:
            batch_inputs, batch_targets = dao.get_batch_train(size=batch_size)
            summary, train_loss, _ = sess.run([merged, loss, optimizer], feed_dict={
                inputs: batch_inputs,
                targets: batch_targets,
            })
            train_writer.add_summary(summary, i)
            if i % 100 == 0:
                batch_inputs, batch_targets = dao.get_batch_test(batch_size)
                summary, test_loss, = sess.run([merged, loss], feed_dict={
                    inputs: batch_inputs,
                    targets: batch_targets
                })
                test_writer.add_summary(summary, i)
                print("Iteration {i}, test loss {mse}, epoch {epoch}".format(i=i, mse=test_loss, epoch=dao.epochs))
            if i % 500 == 0 or dao.epochs >= num_epochs:
                # Make predictions on target data, store in a dictionary, and save to csv
                print("Making predictions on target data.")
                predictions = {}
                for chunk in target_dao.get_data_chunked(size=batch_size):
                    outputs_ = sess.run(outputs, feed_dict={
                        inputs: chunk,
                    })
                    for mtx, output in zip(chunk, outputs_):
                        seq = seq_from_matrix(mtx)
                        predictions[seq] = output
                out_path = "./data/predictions/binary_{}_{}.csv".format(function, taxon_id)
                print("Saving predictions in {out_path}".format(out_path=out_path))
                with open(out_path, "w") as outfile:
                    for seq, preds in predictions.items():
                        outfile.write(seq + "," + ",".join([str(x) for x in preds]) + "\n")
            i += 1