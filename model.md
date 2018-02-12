# CNN网络模型



---

**终于到了最核心的部分，卷积神经网络的构建，通常应用中需要构建合适的神经网络以达到特定的需求，在这里我是需要构建一个卷积神经网络来做简单的人脸识别，下面我来简单介绍一下卷积神经网络：**
>CNN(Convolutional Neural Network)主要用来识别位移、缩放及其他形式的扭曲不变性的二维图形。由于CNN的特征检测层通过训练数据进行学习，所以在使用CNN时，避免了显示的特征抽取，而隐式地从训练数据中进行学习；再者由于同一特征映射面上的神经元权值相同，所以网络可以并行学习，这也是卷积网络相对于神经元彼此相连网络的一大优势。卷积神经网络以其局部权值共享的特殊结构在语音识别和图像处理方面有着独特的优越性，其布局更接近于实际的生物神经网络，权值共享降低了网络的复杂性，特别是多维输入向量的图像可以直接输入网络这一特点避免了特征在提取和分类过程中数据重建的复杂度。通俗来讲，就是通过对卷积神经网络的训练能够得到合适的卷积核，这些卷积核在面对一些特定的图像特征是会产生比较高的输出，以激活特定的神经元，达到能够使神经网络做出最佳判断的目的，权值共享是指卷积核从某小块样本学习到的一些特征应用到图像的其他任意地方去，于是可以达到在图像的不同位置也能得到不同特征的激活值效果。

**下面是此模型的输入输出图表：**

|layer|size-in|size-out|kernel|
|:-:|:-:|:-:|:-:|
|conv_1|64 * 64 * 3|64 * 64 * 32|3 * 3 * 3|
|max_pooling_1|64 * 64 * 32|32 * 32 * 32|2 * 2 * 32|
|conv_2|32 * 32 * 32|32 * 32 * 64|3 * 3 * 32|
|max_pooling_2|32 * 32 * 64|32 * 32 * 64|1 * 1 * 32|
|conv_3|32 * 32 * 64|32 * 32 * 64|3 * 3 * 64|
|max_pooling_3|32 * 32 * 64|32 * 32 * 64|1 * 1 * 32|
|full_connection_1|32 * 32 * 64|128||
|full_connection_2|128|128||
|softmax_linear|128|2||

**下面介绍一下卷积`con2d()`池化`max_pooling()`函数的使用方法：**

### 卷积函数：

```python
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
```

**input:**
>指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一,这就是为什么数据进入第一层卷积层时是64*64*3了(为了图表清楚省略了batch，这里batch=64)其中64*64是图片的分辨率，3是指图片的三个通道(R,G,B)。

**filter:**
>相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，需要注意的是，此参数第三维in_channels相对于input参数的第四维。

**strides:**
>图像在进行卷积操作时的跨度，参数strides使得卷积核可以跳过图像中的一些像素，从而在输出中不包含它们。它可以显著降低输出的维数，降维通常可减少所需的运算量，并且可以避免一些完全重叠的感受域。它具有[batch_size_strides,height_size_strides,width_size_strides,channels_strides],具体含义是[图片批次数据的跳过，图片高的跳过，图片宽的跳过，图片通道的跳过]。通常我们不会修改第一个和第四个参数，那样会导致训练数据的跳过，当我们需要降低输入的维数时通常修改第一个和第三个参数。

**padding:**
>string类型的变量，只能是‘SAME’，‘VALID’它代表不同的卷积方式；用SAME时，卷积的输入和输出尺寸相同，缺失的像素将用零补充，通常卷积核扫过的像素将超过图像的实践像素数；用VALID时，在计算卷积核如何在图像上如何跨越时，需要考虑滤波器的尺寸。这会使得卷积核尽量不越过图片的边界，某些情形下，可能边界也会被填充。

**use_cudnn_on_gpu:**
>bool类型的数据，表示是否使用cudnn加速(GPU加速)，默认为True

### 池化函数

```python
tf.nn.max_pool_with_argmax(input, ksize, strides, padding ,Targmax=None, name=None)
```
**input:**
>一般为卷积后的 feature map，参数格式与上文卷积的输入要求一致，形状为[batch, height, width, channels]

**ksize:**
>池化窗口的大小，与上文卷积函数的filter参数的需求一致，形状为[1, height, weight, 1]，通常不改变第一个和第四个参数，那样可能会导致关键数据的跳过。

**strides:**
>窗口在每一个维度上滑动的步长，依旧与上文卷积函数的strides参数需求一致，形状为[1, stride, stride, 1]，通常一样的不修改参数一和参数四。

**padding:**
>窗口在进行池化操作时的跨度，和卷积操作类似，也是bool类型的值，有‘SAME’，‘VALID’两种不同的池化方式，可参见上面对卷积函数中Padding参数的解释。

### 下面是对Model.py中其他定义函数的作用的解释:

```python
def loss(logits, labels):
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='c_entropy_per_example')
        losses = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('Loss', losses)
    return losses
```

代码构建的`loss()`函数用于计算数据经过神经网络后产生的误差(**后续的优化就是用优化算法改变神经网络中各个激活函数的参数来达到减少此函数所返回的误差**)传入数据是神经网络`inference()`最后softmax层返回结果(**一个列表，其中包含两个数值，分别代表传入数据是类别一还是类别二的概率**)，下面介绍`loss()`函数中所用tensorflow函数：

```python
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='c_entropy_per_example')

```

    第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batch_size，num_classes]，单样本的话，大小就是num_classes
    第二个参数labels：标记着真确结果的标签，大小同上
    函数的意义是对softmax的输出向量[y1,y2,y3...]和样本的正确的标签做一个交叉熵(判断两个分布的相似性)它与tf.nn.softmax_cross_entropy_with_logits()函数的区别就是，它会对样本标签的稀疏表示(独热码)，因为图片标签通常是稀疏的，故本例函数用的要较多一些

```python
losses = tf.reduce_mean(cross_entropy, name='loss')
tf.summary.scalar('Loss', losses)
```

    因为tf.nn.sparse_softmax_cross_entropy_with_logits()函数返回的是一个向量，故还需要使用tf.reduce_mean()函数进行求和的平均操作，最终的tf.summary.scalar()函数是对loss数据进行汇总，方便用tensorboard进行查看

```python
def training(loss, learning_rate):
    with tf.name_scope('Training'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    return train_op
```

代码构建的`training()`函数用于对最终误差的优化，当神经网络返回与正确标签相反的结果，即错误结果时，会导致交叉熵的变大，而本例所用的Adam优化算法，可以找到全局最优解，然后根据最优解反向传播修改神经网络中的参数，使其能返回正确的结果；函数中声明的全局计数变量global_step用于记下训练的步数，其中的传入的参数learning_rate表示`学习速率`，即在进行优化求导的时候每一步的大小

```python
def evaluation(logits, labels):
    with tf.name_scope('Evaluation'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('Accuracy', accuracy)
    return accuracy
```
代码构建的`evaluation()`函数的作用就是对结果的评估；其中主要作用函数是`tf.nn.in_top_k()` 下面是对函数的简单介绍：

```python
in_top_k(predictions, targets, k, name=None)
```
    函数有如下参数:
    predictions：预测的结果
    targets：标有真确结果的标签，大小为样本数
    k：每个样本的预测结果的前k个最大的数里面是否包含targets预测中的标签，一般都是取1，即取预测最大概率的索引与标签对比
**函数使用的示例:**
```python
import tensorflow as tf

input = tf.constant([[0,1],[1,0],[0,1]], tf.float32)  # 假设一个神经网络输出的结果
output = tf.nn.in_top_k(input, [1,1,1], 1)
correct = tf.cast(output, tf.float16)
accuracy = tf.reduce_mean(correct)
with tf.Session() as sess:
    print('模拟神经网络的返回值：\n',sess.run(input))
    print('----------分割线----------')
    print('in_top_k()函数的返回值:\n',sess.run(output))
    print('强制转换后的向量：\n',sess.run(correct))
    print('对三张图片判断的准确率为：\n',sess.run(accuracy))
'''
运行结果:
模拟神经网络的返回值：
 [[ 0.  1.]
 [ 1.  0.]
 [ 0.  1.]]
----------分割线----------
in_top_k()函数的返回值:
 [ True False  True]
强制转换后的向量：
 [ 1.  0.  1.]
对三张图片判断的准确率为：
 0.6665
 '''
```

在示例中，我们假设传入了神经网络`inference()`三张图片，这三张图片都是属于类别二的(标签为1)，然后经过神经网络的运算，输出了一个**预测列表[[0, 1], [1, 0], [0, 1]]** (在这里我假设神经网络能100%确定输入的图片是哪一类)，然后将这个输出结果当作predictions参数，图片的正确**分类标签[1, 1, 1]**当作target参数，一并输入到`in_top_k()`函数中，函数会返回一个判断结果的列表，如果传入的三张图片正好是能认为是第二类别，那么将会返回三个true ，因为第二张图神经网络判断错误，故返回了一个false,这时用`tf.cast()`函数强制将bool类型的值转换为tf.float16的浮点小数值，再用`tf.reduce_mean()`对此向量进行求平均值，就可以得出神经网络的对图片判断的正确率了。
