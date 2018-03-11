# -*- codeing: utf-8 -*-
# 模型的使用
# tensorflow:1.4.0 by fanzhe
import sys
import cv2
import dlib
import numpy as np
from PIL import Image
import tensorflow as tf
import model


# 单个输入文件用于测试(通常用于测试集的验证) #
def img_input(file_path):
    img = Image.open(file_path)
    imag = img.resize([64, 64])
    cv2.imshow('Test',imag)
    image = np.array(imag)
    return image


def put_text(img, index):
    text = target_name[index] + str(round(prediction[:, index][0] * 100, 2)) + '%'
    print(text)
    img = cv2.putText(img, text, (x2, x1), font, 0.7, (255, 255, 255), 2)
    return img

# 测试图片 #
def evaluate_image(image_array, model_dir):
    with tf.Graph().as_default():
        batch_size = 1  # 一次传入一张图片
        n_classes = 4  # 最后分类的结果有多少种 #
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 64, 64, 3])
        dropout_half = tf.placeholder(tf.float32)
        dropout_quart = tf.placeholder(tf.float32)
        logit = model.inference(image,batch_size, n_classes, dropout_half, dropout_quart)
        logit = tf.nn.softmax(logit)
        x = tf.placeholder(tf.float32, shape=[64, 64, 3])

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_dir)  # 读取保存点
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found...')
                exit(0)
            # 在测试模型时要禁用掉dropout
            prediction = sess.run(logit, feed_dict={x: image_array, dropout_half: 1.0, dropout_quart: 1.0})
            return prediction, global_step


if __name__ == '__main__':
    video_dir = 'E:/data/video_data/ssdgyfz.mp4'  # 需要进行人脸识别的视频
    model_dir = 'E:/models/model_3_11'  # 模型调用路径
    target_name =  ['shanxia', 'gonqi','zhushou','faces_other' ]  # 最后识别出来的标签1,2,3,4
    out = cv2.VideoWriter('out1.avi', -1, 10.0, (1280, 720))  # 注:在此修改视频保存尺寸
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    while True:
        print('--------------------------------------------------------')
        print('-                1 .开始摄像人脸识别                   -')
        print('-                2 .开始视频人脸识别                   -')
        print('-                3 .视频人脸识别并保存识别结果          -')
        print('-                4 .退出程序                           -')
        print('--------------------------------------------------------')
        detector = dlib.get_frontal_face_detector()
        size = 64
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
        while (ret==0):
            _, img = cam.read()
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_image, 1)
            if not len(dets):
                cv2.imshow('No face ... Detecting...', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
                    sys.exit(0)
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                face = cv2.resize(face, (size, size))  # 调整图片的尺寸
                b, g, r = cv2.split(face)
                face = cv2.merge([r, g, b])  # 重点！！！
                prediction, global_step = evaluate_image(face, model_dir)  # 对结果进行评估
                print('Model read success! global_step is :%s' % global_step)
                print(prediction)
                max_index = np.argmax(prediction)  # 取结果中最大的
                cv2.rectangle(img, (x2, x1), (y2, y1), (0, 255, 0), 3)  # opencv里的BGR
                img = put_text(img, max_index)
                if cmd == '3':
                    out.write(img)
                    cv2.imshow('frame', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        ret += 1

                else:
                    cv2.imshow('There have some face', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
                        ret += 1 # 用于跳出本次循环

        out.release()
        cv2.destroyAllWindows()
        break
    sys.exit(0)
