# -*- codeing: utf-8 -*-
import sys
import os
import cv2
import dlib

# 下载 lfw.tgz 并解压所有文件到../input_img

input_dir = '../data/input_img'
output_dir = '../data/other_faces_64'
pic_num = 2000
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
detector = dlib.get_frontal_face_detector()  # 使用dlib自带的frontal_face_detector作为我们的特征提取器

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            img_path = path + '/' + filename
            img = cv2.imread(img_path)   # 从文件读取图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图像灰化，降低计算复杂度
            dets = detector(gray_img, 1)  # 使用detector进行人脸检测 dets为返回的结果
            print('Being processed picture {},{} faces dectected!'.format(index, len(dets))  
            for i, d in enumerate(dets):  # 此处是从人脸图片中提前人脸，故无需再进行人脸判断
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]   # 定位并剪出人脸

                # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
                face = relight(face, random.uniform(0.7, 1.3), random.randint(-20, 20))
                face = cv2.resize(face, (size, size))  # 将图片改成特定分辨率
                cv2.imshow('image', face)
                cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)  # 保存图片
                index += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键中断
                sys.exit(0)
            if index == pic_num:
                print('Done!')
                sys.exit(0)
