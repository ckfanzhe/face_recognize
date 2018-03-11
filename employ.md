# 对训练好的模型的运用

---

终于，到了最能直观表现卷积神经网络魅力的时候了，`employ.py`代码文件就是对之前训练好的神经网络模型的运用，它会对卷积神经网络的Tensorflow流图进行复现，然后传入单张或多张图片进行评估，最终返回预测值，预测值将会是一个向量，对预测值进行判断提取分析，就可以得到最终分类的结果了

**所用到的Python库及版本：**

    Tenosrflow-gpu = 1.4.0
    Opencv-python = 3.3.0
    Numpy = 1.13.3
    Pillow = 5.0.0
    Dlib = 19.8.1
    model(自己编写的神经网络模型文件)
    sys
    
### 代码解析(`employ.py`):

```python
# 单个输入文件用于测试(通常用于测试集的验证) #
def img_input(file_path):
    img = Image.open(file_path)  # 打开指定路径的文件
    imag = img.resize([64, 64])  # 修改图片的宽高
    cv2.imshow('Test',imag)
    image = np.array(imag)  # 将图片转换成矩阵向量
    return image
```

上述代码定义了一个`img_input()`函数，函数的作用是将输入的单张图片路径所指向的图片载入到内存中，然后将它改成特定的分辨率，然后再转换成向量形式，方便传入神经网络进行评估。此函数用于单张图片的预测，如果需要知道模型在测试数据集上的分类准确率，可以使用此函数对图片数据进行导入验证

```python
def put_text(img, index):
    text = target_name[index] + str(round(prediction[:, index][0] * 100, 2)) + '%'
    print(text)
    img = cv2.putText(img, text, (x2, x1), font, 0.7, (255, 255, 255), 2)
    return img
```
此函数的作用是将神经网络分类的结果进行处理后，打印在单张图片上，其中使用了Opencv中的`putText()`函数，函数传入参数的顺序是：图片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细 ，可以通过对参数的修改来达到使图片上输出不同样式结果的目的；外部函数`put_text()`的两个参数：一个是图片数据，另一个是神经网络最终返回向量值中最大的索引，用于在图片上输出最终图片辨认的正确结果。

```python
# 测试图片 #
def evaluate_image(image_array, model_dir):
    with tf.Graph().as_default():
        batch_size = 1  # 一次传入一张图片进行评估
        n_classes = 4  # 最后分类的结果有多少种
        image = tf.cast(image_array, tf.float32)  # 强制转换函数，将数据强制转换成其他的类型
        image = tf.image.per_image_standardization(image)  # 图片的标准化
        image = tf.reshape(image, [1, 64, 64, 3])  # 将图片进行转换
        dropout_half = tf.placeholder(tf.float32)  # 设置一个tf.float32类型的占位符
        dropout_quart = tf.placeholder(tf.float32)
        logit = model.inference(image,batch_size, n_classes, dropout_half, dropout_quart)
        logit = tf.nn.softmax(logit)  # 对最终的结果进行softmax函数操作(返回和等于一的概率向量)
        x = tf.placeholder(tf.float32, shape=[64, 64, 3])  # 创建图片占位变量
        
        saver = tf.train.Saver()  # 创建模型读取对象

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_dir)  # 读取保存点
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)  # 对模型数据进行读取
                print('Loading success, global_step is %s' % global_step)  # 打印出导入模型的最终训练数
             else:
                print('No checkpoint file found...')
                sys.exit(0)
            # 在测试模型时要禁用掉dropout，将其值设置为1，代表保留全部神经元
            prediction = sess.run(logit, feed_dict={x: image_array, dropout_half: 1.0, dropout_quart: 1.0})  # 将数据填入Tensorflow数据流图中进行判断
            return prediction, global_step
```
上述代码定义了一个`evaluate_image()`的函数，它的作用是将之前构建的Tensorflow流图进行复现，然后导入之前训练好的模型，将单张图片的矩阵向量转换成能被神经网络接受的向量格式，然后传入恢复了模型数据的Tensorflow流图中，进行评估，最终返回神经网络的预测值，和模型最终的训练步数。
其中主要用`model.inference()`对传入的单张图片进行评估，用`tf.train_checkpoint_state()`读取最终的模型保存点(通常读取最后一步的模型数据)，用`saver.restore()`函数载入模型中的神经网络参数(这里即是训练好的参数)，最后进行结果的预测；在这里要注意的是dropout的参数要设置为1(即不使用dropout),dropout的作用是使一定概率的神经元随机失活以提高神经网络模型的泛化性，避免过拟合现象。

```python
if __name__ == '__main__':
    video_dir = 'E:/data/video_data/ssdgyfz.mp4'  # 需要进行人脸识别的视频
    model_dir = 'E:/models/model_3_11'  # 模型调用路径
    target_name =  ['shanxia', 'gonqi','zhushou','faces_other' ]  # 最后识别出来的标签1,2,3,4
    out = cv2.VideoWriter('out1.avi', -1, 10.0, (1280, 720))  # 注:在此修改视频保存尺寸
    font = cv2.FONT_HERSHEY_SIMPLEX  # 在图片上打印输出结果时使用默认字体
    size = 64 # 定义传入神经网络训练所用图片的尺寸
    while True:
        print('--------------------------------------------------------')
        print('-                1 .开始摄像人脸识别                   -')
        print('-                2 .开始视频人脸识别                   -')
        print('-                3 .视频人脸识别并保存识别结果          -')
        print('-                4 .退出程序                           -')
        print('--------------------------------------------------------')
        detector = dlib.get_frontal_face_detector()
        cmd = str(input("........>"))
        if cmd == '1':
            cam = cv2.VideoCapture(0)
        elif (cmd == '2')or(cmd == '3' ):
            cam = cv2.VideoCapture(video_dir)
        elif cmd == '4':
            break
        else:
            print("请在上述的选项中进行输入！")
            continue
        ret = 0
        while (ret==0):  # 设置一个结束标准，当循环需要结束时，只需将ret值进行修改即可
            _, img = cam.read()
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_image, 1)
            if not len(dets):
                cv2.imshow('No face ... Detecting...', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
                    ret+=1 # 当按下q时改变while循环的true值
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]  # 对整张图片中的人脸进行裁剪
                face = cv2.resize(face, (size, size))  # 调整图片的尺寸
                b, g, r = cv2.split(face)
                face = cv2.merge([r, g, b])  # 重点！！！
                prediction, global_step = evaluate_image(face, model_dir)  # 对结果进行评估
                print('Model read success! global_step is :%s' % global_step)
                print(prediction)  # 打印出当前预测值，方便进行错误调试
                max_index = np.argmax(prediction)  # 取结果中最大的下标
                cv2.rectangle(img, (x2, x1), (y2, y1), (0, 255, 0), 3)  # opencv里的BGR顺序，这里为绘制一个绿色的边框
                img = put_text(img, max_index)
                if cmd == '3':
                    out.write(img)  # 将图片的每一帧写入到视频中
                    cv2.imshow('frame', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        ret += 1

                else:
                    cv2.imshow('There have some face', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
                        ret += 1 # 用于跳出本次循环

        out.release()  # 释放视频资源
        cv2.destroyAllWindows() # 销毁所有窗口
        break
    sys.exit(0)
```

上述代码用了嵌套的while循环，外面的循环用于判断所要执行的命令，里面的循环用来取视频的每一帧图像，并对图像进行编辑，其中人脸检测的方式在`data_preparation/faces_catch.py`中有介绍，通过人脸检测可以取到人脸在图像中的位置，然后用`cv2.rectangle()`函数进行矩形绘制，然后用自定义函数`put_text()`在特定位置输出文字；其中人脸辨别的方式是：将识别到的人脸单独的裁剪剪出来，然后传入恢复好的Tensorflow神经网络流图中进行运算，得到输出结果，通过对结果的简单处理就可以输出到图片上。这里要注意的是，图片数据经过opencv处理后的图片矩阵是按照BGR顺序进行排序的，而我们训练神经网络的图片矩阵是RGB方式，图片在放入神经网络预测时，需要将图片所转换成的矩阵三维顺序进行变换为RGB顺序的(我在这爬了好久的坑，之前预测结果一直不对，后面一步一步调试输出才找出问题)；还有对预测后的视频进行保存时记得修改成传入视频尺寸(后续我会更新一下，让它自动修改)，不然会出现保存视频尺寸不对称的错误
