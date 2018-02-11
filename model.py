import tensorflow as tf


def inference(images,batch_size, n_classes, dropout_half, dropout_quart):
    with tf.name_scope('conv_1'):
        with tf.variable_scope('Weight_Biases1'):
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], stddev=1.0, dtype=tf.float32),
                                  name='weights_1', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[32]),
                                 name='biases_1', dtype=tf.float32)

        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name='conv_1')

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling_1')

    with tf.name_scope('Conv_2'):
        with tf.variable_scope('Weight_Biases2'):
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=0.1, dtype=tf.float32),
                                  name='weights_2', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                                 name='biases_2', dtype=tf.float32)

        conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv_2')

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling_2')

    with tf.name_scope('Conv_3'):
        with tf.variable_scope('Weights_Biases3'):
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3 , 64, 64], stddev=0.1, dtype=tf.float32),
                                  name='weights_3', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32,shape=[64]),
                                 name='biases_3', dtype=tf.float32)
        conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv_3')

        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling_3')

    with tf.name_scope('Full_connect_1'):
        with tf.name_scope('Reshape'):
            reshape = tf.reshape(pool3, shape=[batch_size, -1])
            dim = reshape.get_shape()[1].value
        with tf.variable_scope('Weight_Biases_1'):
            weights = tf.Variable(tf.truncated_normal(shape=[dim, 128], stddev=0.005, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                 name='biases', dtype=tf.float32)
        full_connect_1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='full_con1')

        full_connect_1_drop = tf.nn.dropout(full_connect_1, dropout_half)
    with tf.name_scope('Full_connect_2'):
        with tf.variable_scope('Weight_Biases_2'):
            weights = tf.Variable(tf.truncated_normal(shape=[128, 128], stddev=0.005, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                 name='biases', dtype=tf.float32)

        full_connect_2 = tf.nn.relu(tf.matmul(full_connect_1_drop, weights) + biases, name='full_con2')

        full_connect_2_drop = tf.nn.dropout(full_connect_2, dropout_quart)
    with tf.name_scope('Softmax_linear'):
        with tf.variable_scope('Weight_Biases_s'):
            weights = tf.Variable(tf.truncated_normal(shape=[128, n_classes], stddev=0.005, dtype=tf.float32),
                                  name='softmax_linear', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),
                                 name='biases', dtype=tf.float32)

        softmax_linear = tf.add(tf.matmul(full_connect_2_drop, weights), biases, name='softmax_linear')
    return softmax_linear


def loss(logits, labels):
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                               name='c_entropy_per_example')
        losses = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('Loss', losses)
    return losses


def training(loss, learning_rate):
    with tf.name_scope('Training'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.name_scope('Evaluation'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('Accuracy', accuracy)
    return accuracy
