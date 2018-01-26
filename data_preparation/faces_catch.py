# -*- codeing: utf-8 -*-
# 用于从摄像头或视频中提取人脸数据
import cv2
import dlib
import os
import random

# 从视频中截取的人脸数据，保存到../data/faces_my目录

output_dir = '../data/hei'  # 输出文件夹
video_dir = '../data/video_data/video2.mp4'  # 视频输入路径
size = 64  # 所截取的图片分辨率为64*64
pic_num = 200  # 所截取的图片数量

if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # 如果不存在输出目录则创建


# 改变图片的亮度与对比度 #
def relight(imge, gain=1, bias=0):
    """
    此函数用于改变图片的亮度，对比度
    :param imge: 输入为图片数据
    :param gain: 对比度
    :param bias: 亮度
    :return: 返回图片数据
    """
    w = imge.shape[1]  # 取图片宽度
    h = imge.shape[0]  # 取图片高度
    for k in range(0, w):
        for j in range(0, h):
            for c in range(3):
                temp = int(imge[j, k, c]*gain + bias)  # 以遍历的方式改像素点的值(相当于改亮度、对比度)
                if temp > 255:
                    temp = 255  # 如果结果超过255就设为255
                elif temp < 0:
                    temp = 0
                imge[j, k, c] = temp
    return imge


detector = dlib.get_frontal_face_detector()  # 使用dlib自带的frontal_face_detector作为我们的特征提取器
camera = cv2.VideoCapture(video_dir)  # 打开摄像头 参数为输入流，可以为摄像头或视频文件，当参数为0或1时，将从其他摄像头取数据
index = 1
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
