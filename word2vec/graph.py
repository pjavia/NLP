import tensorflow as tf
import preprocessing
from random import shuffle, seed
from datetime import datetime
from tensorflow.contrib.tensorboard.plugins import projector
import os

g = tf.Graph()
batch = 20
par = (2, 128, 'datasetSentences.txt')
obj = preprocessing.word2vec(par)
obj.fetch_data()
obj.prepare_data_for_word2vec()
vocabulary_size = obj.vocabulary
embedding_size = obj.embeddings_dim


with g.as_default():
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),  name='word_embedding')
    print embeddings, 'embeddings'
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0 / tf.sqrt(float(embedding_size))))
    print nce_weights, 'nce_weights'
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    print nce_biases, 'nce_biases'
    train_inputs = tf.placeholder(tf.int32, shape=[batch])
    train_labels = tf.placeholder(tf.int32, shape=[batch, 1])
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                                         num_sampled=20, num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    saver = tf.train.Saver()
    summary_l = tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()

with tf.Session(graph=g) as sess:
    train_writer = tf.summary.FileWriter('summary_directory')
    data_size = len(obj.numeral_pairs)
    data = obj.numeral_pairs
    sess.run(tf.global_variables_initializer())
    count = 0
    for epoch in range(0, 1):
        for i in range(0, data_size):
            count += 1
            print epoch, '--------->epoch', count, '---------->iteration'
            seed(datetime.now())
            shuffle(data)
            batch_xs = []
            batch_ys = []
            for j in range(0, batch):
                batch_xs.append(data[j][0])
                batch_ys.append([data[j][1]])
            l_, _ = sess.run([loss, optimizer], feed_dict={train_inputs:batch_xs, train_labels:batch_ys})
            print l_, 'loss'
            summary_full = sess.run(summary_op, feed_dict={train_inputs: batch_xs, train_labels: batch_ys})
            train_writer.add_summary(summary_full, count)

            if count == 300:
                break
    saver.save(sess, os.path.join('summary_directory', "model.ckpt"), 1)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.metadata_path = os.path.join('summary_directory', 'metadata.tsv')
    embedding.tensor_name = embeddings.name
    projector.visualize_embeddings(train_writer, config)
    with open("summary_directory/metadata.tsv", "w") as record_file:
        for i in obj.metadata:
            record_file.write("%s\n" % i)

