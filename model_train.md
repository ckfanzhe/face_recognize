# 模型的训练

---

在前面，我们定义了一个卷积神经网络模型，深度学习的其中的一个精髓就是对神经网络的训练，并通过不断的修改神经网络的定义参数，以达到令人满意的准确率，下面我来介绍一下用于神经网络训练的`model_train.py`代码文件：

**所用到的python库版本:**

    tensorflow-gpu = 1.4.0
    numpy = 1.13.3
    os
    model(自己编写的模型代码)
    image_operate(自己编写的图片预处理代码)

### 代码解析(`model_train.py`)：

```python
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

```
正如注释所说，此段代码是对，下面模型训练所用的变量和路径的声明，其中`target_1,target_2`变量要对应读入训练数据的文件夹名，而`train_files`必须要是训练数据存放文件夹的上级目录。因为一次性将要训练的几百张，几千张图片同时加载进内存进行训练对于我们非专业的训练设备是很麻烦的(性能,内存方面的不足)，所以我们是采取分批次训练的方式，一次加载进内存128张图片，其中64张为一批次，放入神经网络中进行训练，当前64张被取出时，后面会立马补上64张，使内存队列中始终能存在有128张。

```python
# 训练文件的准备 #
train, train_label = image_operate.get_files(train_files, target_1, target_2)
train_batch, train_label_batch = image_operate.get_batchs(train, train_label, img_w, img_h, batch_size, capacity)
```

上述代码也如注释所说，是对训练文件的准备。这里调用了自己写的python模块`image_operate.py`中的两个数据处理函数 ，第一个函数`get_files()`能将图片路径，和标签取出并分别放入train,train_label列表中，然后再用`get_batch()`函数对图片进行编辑返回能够直接传入神经网络进行训练的**图片batch**，和与图片batch对应的**标签label_batch** （***详细用法见image_operate.md说明文件***）

```pthon
# 训练操作的定义 #
train_logits = model.inference(train_batch, batch_size, n_classes, dropout_half, dropout_quart)
train_loss = model.loss(train_logits, train_label_batch)
train_op = model.training(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)
```

此代码定义了训练的操作其中`train_logits`变量是一个带有两种类别概率的向量；`train_loss`是一个取交叉熵并求和的误差变量；`train_op`是一种训练方式；`train_acc`是一个神经网络对传入的64张图片进行预测后的正确率

```python
# log汇总 #
summary_op = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()  # 模型的保存
sess.run(tf.global_variables_initializer())

# 启用队列 #
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
```

上述代码的作用是对所有log数据的汇总，以及模型的保存；而启用队列Coordinator类可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常。QueueRunner类用来协调多个工作线程同时将多个张量推入同一个队列中。两者互相配合，以达到多线程的效果

```python
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
```
最后的代码就是真正的训练了，其中最外层使用了try语句来扑捉异常，防止训练被异常中断，用finally语句来最终执行`coord.request_stop()`函数使所有子线程的执行停止；用for循环语句，使得训练操作能够执行`all_step`次，然后用 `step%10 == 0`语句判断是否是10的倍数，如果是则打印当前批次的误差及准确率，最后，每隔save_step步保存一次训练好的模型；通过不断的训练 ，神经网络中的参数对输入图片进行判断后会不断趋向于返回准确的输出结果；卷积层能很好的对图片特征进行提取，而全连接层显著提高神经网络对空间的划分能力(丰富的VC维)，使得它能对复杂的数据进行划分
。