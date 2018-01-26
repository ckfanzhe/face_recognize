# 人脸识别的数据准备
---
当前文件夹的两个文件`faces_catch.py` 和 `faces_other.py` 分别是用来准备自己的人脸数据和它人的人脸数据，模型比较简单，是一个普通的二分类，用于分辨输入的图片是不是特定人物，所以要准备的数据有本人`faces_my`的人脸，和许多的其他`faces_other`人脸,思路是用Opencv的dlib库在输入图片中找出人脸，并修改尺寸进行保存，下面是对代码的一些简单解释：

**所用到的Python库：**

    opencv=3.3.0
    dlib=19.8.1 
    os sys random
### 代码解析(`faces_my.py`):
```python
def relight(imge, gain=1, bias=0):
    w = imge.shape[1]  # 取图片宽度
    h = imge.shape[0]  # 取图片高度
    for k in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(imge[j, k, c]*gain + bias)  # 以遍历的方式修改像素点的值(相当于改亮度、对比度)
                if tmp > 255:
                    tmp = 255  # 如果结果超过255就设为255
                elif tmp < 0:
                    tmp = 0
                imge[j, k, c] = tmp
    return imge
```
定义的`relight()`函数，通过遍历图片三个通道`(RBG)`的所有像素点并通过特定的值`(gain，bias)`改变像素点的值 以达到修改整张图片亮度和对比度的效果
>其中j和k表示像素位于第j行和第k列，c为图片的通道，这个式子可以用来作为我们在opencv中控制图像的亮度和对比度的理论公式(其中gain是对比度，值通常为 0.0-3.0 ；bais为亮度)：
>>img[j ,k , c] = img[j ,k ,c] * gain + bais

```python
detector = dlib.get_frontal_face_detector()  # 使用dlib自带的frontal_face_detector作为我们的人脸提取器
camera = cv2.VideoCapture(video_dir)  # 打开摄像头,参数为输入流，可以为摄像头或视频文件
```
通过改变函数`cv2.VideoCapture()`中参数的值来修改从那里输入视频，如果是一个视频文件路径，那么输入流为视频文件，如果参数是0,1,2之类的数字，那么将从电脑连接的网络摄像头设备取视频作为输入流
```python
while True:
    if index <= pic_num:
        print('Being processed picture %s' % index)
        # 从视频流中读取照片
        success, img = camera.read()
        # 转为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)
        if len(dets) >= 1:
            print("{} faces detected".format(len(dets)))  # 打印出检测到的人脸数
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = img[x1:y1, x2:y2]   # 定位并剪出人脸

            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            face = relight(face, random.uniform(0.7, 1.3), random.randint(-20, 20))
            face = cv2.resize(face, (size, size))  # 将图片改成特定分辨率
            cv2.imshow('Face', face)
            cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
            index += 1  # 循环pic_num次
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键中断
            break
    else:
        print('Finished!')
        break
```

用while True循环确保能取到pic_num数量的代码，在程序运行中可以按下**q键**进行中断；5到9行的代码是从视频流中读取图片用`cv2.cvtColor()`函数将提取的图片转换为灰度图像用于**detector**的人脸检测；10到11行代码是 通过判断dets列表的长度返回检测到的人脸数；12到17行使用`enumerate()` 函数遍历序列中的元素以及它们的下标,其中下标i即为人脸序号,下面是变量for循环中变量的含义：

|变量名|含义|
|:---:|:--:|
|x1|人脸上边距离图片上边界的距离|
|x2|人脸左边距离图片左边界的距离|
|y1|人脸下边距离图片下边界的距离|
|y2|人脸右边距离图片右边界的距离|

>`cv2.waitKey()` 是一个键盘绑定函数。需要指出的是它的时间尺度是毫秒级。函数等待特定的几毫秒，看是否有键盘输入。特定的几毫秒之内，如果按下任意键，这个函数会返回按键的ASCII码值，程序将会继续运行。如果没有键盘输入，返回值为 -1，如果我们设置这个函数的参数为 0，那它将会无限期的等待键盘输入。在此处时判断在等待期间是否按下了**q**键，如果按下则执行if的子语句。

    关于为什么要有0xFF:
    On some systems, waitKey() may return a value that encodes more than just the ASCII keycode.
    On all systems, we can ensure that we extract just the ASCII keycode by reading the lastbyte from 
    the return value like this: 
    keycode = cv2.waitKey(1)
    if keycode != -1: 
        keycode &= 0xFF
### 代码解析(`faces_other.py`):
`faces_other.py` 的做用是将从此处[人脸数据](http://vis-www.cs.umass.edu/lfw/lfw.tgz "人脸数据下载")下载解压后的人脸数据进行剪切(提取出人脸)，然后进行统一
```python
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
```
此段代码的意思是遍历目标文件夹`input_dir`下的每一个文件夹中的每一个图片文件

**其他的部分基本上与上述代码相同，对遍历的每一张图片进行人脸检测并裁剪出特定大小的人脸用作训练数据**
