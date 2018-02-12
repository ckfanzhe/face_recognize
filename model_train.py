import os
import image_operate
import model
import numpy as np
import tensorflow as tf

# 变量声明 #
n_classes = 2  # 最终神经网络的两个分类
img_w = 64  # 图片的宽
img_h = 64  # 图片的高
batch_size = 64  # 一个批次的数据量,同时也是全连接层第一层的shape
capacity = 128   # 图片输入队列的大小
save_step = 100  # 每隔多少次保存
all_step = 300   # 训练多少次
learning_rate = 0.0001  # 学习速率
target_1 = 'hei'  # 存放分类图片的文件夹名,对应target_a
target_2 = 'faces_other'  # 对于target_b
dropout_half = tf.placeholder(tf.float32)  # 定义dropout值的类型
dropout_quart = tf.placeholder(tf.float32)

# 训练文件路径的声明 #
train_files = 'E:/rootcoding/tensorflow/face_recognize/data'  # 训练样本的读入路径
logs_train_dir = 'E:/models/model_2_11_2'  # logs,model存储路径

# 训练文件的准备 #
train, train_label = image_operate.get_files(train_files, target_1, target_2)
train_batch, train_label_batch = image_operate.get_batchs(train, train_label, img_w, img_h, batch_size, capacity)

# 训练操作的定义 #
train_logits = model.inference(train_batch, batch_size, n_classes, dropout_half, dropout_quart)
train_loss = model.loss(train_logits, train_label_batch)
train_op = model.training(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)

# log汇总 #
summary_op = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()  # 模型的保存
sess.run(tf.global_variables_initializer())

# 启用队列 #
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 开始训练 #
try:
    # 执行all_step步的训练，一步一个batch
    for step in np.arange(all_step):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
                                        feed_dict={dropout_half: 0.5, dropout_quart: 0.75})
        # 每隔10步打印一次当前的误差及准确率
        if step % 10 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            summary_str = sess.run(summary_op, feed_dict={dropout_half: 0.5, dropout_quart: 0.75})
            train_writer.add_summary(summary_str, step)
        # 每隔save_step步，保存一次训练好的模型
        if step % save_step == 0:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            print('Model Save Success! step: %s' % step)

except tf.errors.OutOfRangeError:
    print('Training done !')

finally:
    coord.request_stop()


