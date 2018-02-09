import os
import numpy as np
import tensorflow as tf

# 下面设置训练要用的分类类型
mfaces = []
label_mfaces = []
ofaces = []
label_ofaces = []
target_1 = 'hei'  # 存放分类图片的文件夹名,对应target_a
target_2 = 'faces_other'  # target_b

# 对文件进行打乱 #
def shuffle(list_1, list_2, label_1, label_2):
    image_list = np.hstack((list_1, list_2))  # 将两个列表中的元素整合成一个列表
    label_list = np.hstack((label_1, label_2))  # 同上
    temp = np.array([image_list, label_list])  # 将数据转换为二维矩阵
    temp = temp.transpose()  # 将二维矩阵进行转置
    np.random.shuffle(temp)  # 对数据进行随机打乱
    all_image_list = list(temp[:, 0])  # 取图片路径
    all_label_list = list(temp[:, 1])  # 取图片下标
    n_sample = len(all_label_list)  # 取列表的长度
    tra_images = all_image_list[0:n_sample]  # 取图片路径元素个数为前n_sample个
    tra_labels = all_label_list[0:n_sample]  # 取下标元素个数也为n_sample个
    tra_labels = [int(float(i)) for i in tra_labels]  # 将列表中的数字强制转换为int型(整形)
    return tra_images,tra_labels

# 下面是获取文件名并将其导入上述声明的列表 #
def get_files(train_files, target_a, target_b):
    for file in os.listdir(train_files + '/' + target_a):
        mfaces.append(train_files + '/' + target_a + '/' + file)
        label_mfaces.append(0)
    for file in os.listdir(train_files + '/' + target_b):
        ofaces.append(train_files + '/' + target_b + '/' + file)
        label_ofaces.append(1)
    return shuffle(mfaces, ofaces, label_mfaces, label_ofaces)


def get_batchs(image, label, image_w, image_h, batch_size, capacity):
    # 路径，下标数据强制转换为适合的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    with tf.name_scope('image_input_queue'):
        # 创建输入队列并从队列中读取图像
        input_queue = tf.train.slice_input_producer([image, label])
        label = input_queue[1]
        image_contents = tf.read_file(input_queue[0])
    with tf.name_scope('image_operate'):
        # 对图片进行解码
        image = tf.image.decode_jpeg(image_contents, channels=3)

        #  数据预处理：裁剪或填充(大于指定尺寸[image_w,image_h]裁剪，小于填充)
        image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h)
        image = tf.image.per_image_standardization(image)  # 对所有图片进行标准化操作

    # 生成batch
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)

    # 重新排列label，行数为batch_size
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    print('Face_Image prepare successful!')
    return image_batch, label_batch
