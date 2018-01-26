# -*- codeing: utf-8 -*-
import sys
import os
import cv2
import dlib

# 下载 lfw.tgz 并解压所有文件到../input_img
# 图片获取方式命令行下：wget http://vis-www.cs.umass.edu/lfw/lfw.tgz

input_dir = '../data/input_img'
output_dir = '../data/other_faces_128'
pic_num = 2000
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
detector = dlib.get_frontal_face_detector()  # 使用dlib自带的frontal_face_detector作为我们的特征提取器

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path + '/' + filename
            img = cv2.imread(img_path)   # 从文件读取图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图片
            dets = detector(gray_img, 1)  # 使用detector进行人脸检测 dets为返回的结果
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                face = cv2.resize(face, (size, size))  # 调整图片的尺寸
                cv2.imshow('image', face)
                cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)  # 保存图片
                index += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键中断
                sys.exit(0)
            if index == pic_num:
                print('Done!')
                sys.exit(0)