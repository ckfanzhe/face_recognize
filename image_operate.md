# 数据在训练前的准备

---

文件`data_preparation.py`的做用是读取当前已经准备好的数据，进行随机打乱，剪切，填补，归一化等操作后按照一定数量的批次传入神经网络进行训练，下面是对代码的一些简单解释(初学,所有对Tensorflow的理解都来自参考网上博客和官方文档，如有错误欢迎指正)：

**所用到的Python库**

    Tensorflow-gpu = 1.4.0
    numpy = 1.13.3
    os
### 代码解析(`data_preparation.py`)
```python
faces_image = []
faces_label = []

```
代码的含义是声明`faces_image,faces_label`为两个列表，分别用于存放图片地址信息和图片标签信息，标签列表中的数据是和图片列表中的图片地址相对应的，如果有三个类别，那么将会对照图片地址生成相对应的标签顺序如0、1,、0、2，两个列表将用于接下来的图片顺序打乱（**在一个批次的图片数据传入神经网络进行训练时，如果全部是某一类数据，而下一批次传入的是另一类数据，那么将可能会对神经网络训练产生影响**）。
```python
def get_files(train_files, target=[]):
    for i,f in enumerate(target):
        faces_image.append([train_files + '/' + target[i] + '/' + file for file in os.listdir(train_files + '/' + target[i])])  # 直接用列表表达式创建列表文件数据
        faces_label.append([i for c in os.listdir(train_files + '/' + target[i])])

    return shuffle(faces_image, faces_label)
```
代码定义了一个`get_files()`的函数，该函数的作用是遍历**train_files** + 目标文件夹**target** 列表中组成的绝对路径下的所有文件，并用列表表达式将其添加到先前定义好的列表中，方便接下来的打乱；函数传入的两个参数 `train_files,target`分别是训练数据的前一级绝对路径、包含目标图片数据文件夹名称的列表，其中图片数据一定要储存到相应的目标图片数据文件夹里；函数返回是通过`shuffle()`函数打乱顺序的图片数据和图片标签,如果需要对更多目标的人脸数据进行识别，在上面声明更多的列表，并修改一下**target**列表即可（注：列表中的目标文件夹名字需要对应训练数据的存放文件夹）。
```python
def shuffle(image_data, label_data):
    image_list = []
    label_list = []
    for i in image_data:  # 将所有图片地址文件列表合并
        image_list = np.hstack((image_list, i))
    for i in label_data:  # 将标签文件列表合并
        label_list = np.hstack((label_list, i))
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
```
代码定义了一个可以将传入的列表数据进行组合打乱的函数，下面是打乱后的结果（注：打乱只是打乱排列顺序，并不会打乱标签间的对应关系）：
>['E:/rootcoding/tensorflow/face_recognize/data/faces_other/12.jpg', 'E:/rootcoding/tensorflow/face_recognize/data/hei/119.jpg', 'E:/rootcoding/tensorflow/face_recognize/data/faces_other/1236.jpg', 'E:/rootcoding/tensorflow/face_recognize/data/faces_other/590.jpg', 'E:/rootcoding/tensorflow/face_recognize/data/faces_other/819.jpg']
[1, 0, 1, 1, 1]

**其中零是对应目标一的图片数据，一是对应目标二的数据，在重组随机后，可以将四个列表合并成两个，并且对图片的顺序进行随机的打乱**。
```python
def get_batchs(image, label, image_w, image_h, batch_size, capacity):
    # 将路径，下标数据强制转换为适合的格式
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

        # 数据预处理：裁剪或填充(大于指定尺寸[image_w,image_h]裁剪，小于填充)，此处可以省略，因为预先准备的数据是固定尺寸的[64 ,64]
        image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h)
        image = tf.image.per_image_standardization(image)  # 对所有图片进行归一化操作

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
```
现在终于开始接触到Tensorflow了，上述代码的的作用是先将传入的图片路径，下标强制转换为tensorflow能够接受的格式，然后创建一个输入队列并从队列中读取图像，再对图片进行解码，裁剪或填充，归一操作，然后生成特定大小的batch，函数的返回即可放入神经网络进行训练；下面是对其中涉及到的tensorflow操作的一些简单说明(详情请见官方文档：[Tensorflow中文社区](http://www.tensorfly.cn/ "跳转至中文社区"))：

|Tensorflow函数|作用|
|:---:|:--:|
|tf.cast()|类型转换函数,将Python数据转换为TF所需要的数据|
|tf.name_scope()|通过区域名的划分,使得tensorboard显示出的流图可读性更好|
|tf.train.slice_input_producer()|对两个列表进行切片(将两个列表内容进行合并)|
|tf.image.decode_jpeg()|对jpeg格式的图片数据进行解码(后续操作的基础)|
|tf.image.resize_image_with_crop_or_pad()|对图片进行操作(大于指定尺寸[image_w,image_h]裁剪，小于填充)|
|tf.image.per_image_standardization()|此函数的运算过程是将整幅图片标准化（不是归一化)，加速神经网络的训练|
|tf.train.batch()|按顺序读取数据，且队列中的数据始终是一个有序的队列|
|tf.reshape()|按照要求更改数组，矩阵的格式|

