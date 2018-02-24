
### 简单介绍入坑经历：
>入深度学习和Tensorflow的坑已经有几个月的时间了,从买下第一本相关书籍《面向机器智能的Tensorflow实践》到写出这个简单的Demo,其中爬了太多的坑。那时我的显卡还不支持cuda、cudnn加速,所以只能用cpu进行网络训练(灰常慢,还好跑的示例简单),因为Tensorflow还算是比较流行,所以网上有许多可以查看的文档和代码,让我从什么都不知道到现在勉强摸索出一些门路,从当初的书上第一个线性回归的简单示例到现在能做一些十分简单的图像识别任务,爬了这么多坑让我深刻的意识到系统学习的重要性。依旧还记得在运行书上的示例代码的时候,满屏的红字(Error)真的是让人崩溃,于是,我又得老实的去请教度娘。这个Demo很大程度上参照了2012年ImageNet的竞赛中脱颖而出的AlexNet,它分为数据准备与导入、模型、模型训练、模型应用等文件；但因为训练网络的数据量很少,所以不需要像AlexNet那样的多个GPU加速计算,Demo中模型是三层卷积池化层加两层全连接层加最终的回归层(详细参数见model.md说明文件),其中的代码,函数作用都参照了许多博客。不足的是,没有神经网络在验证数据集上的验证,只有直接配合Opencv的应用。

### 下面是源代码以及备注文件的说明：

|代码或文件|说明|
|:---:|:---:|
|data_preparation/faces_catch.py|对自己或目标人脸数据的准备(源代码)|
|data_preparation/faces_other.py|对其他人脸数据的准备(源代码)|
|data_preparation/README.md|对上述两个文件的代码说明(markdown说明文件)|
|image_operate.py|对传入神经网络的数据进行预处理(源代码)|
|image_operate.md|对image_operate.py的代码说明(markdown说明文件)|
|model.py|卷积神经网络的模型(源代码)|
|model.md|对卷积神经网络模型的说明(markdown说明文件)|
|model_train.py|对模型进行训练(源代码)|
|model_train.md|对模型训练代码的解释(markdown说明文件)|
|employ.py|对模型的运用(源代码)|
|employ.md|模型运用源代码解释(markdown说明文件)|



