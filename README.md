# pytorch_learning_my_face
这个项目是我学习深度学习的第一个小玩具，通过pytorch进行CNN学习，识别出自己和别人的脸。

##环境要求
>OS：Ubuntu16.04LTS  
>python：3.6.3  
>opencv：3.3.1  
>pytorch：0.3

## 获取素材
首先需要准备好图片素材来给我们的模型进行训练。从该[网站](http://vis-www.cs.umass.edu/lfw/) 别人的人脸照片，然后运行

```shell
python get_pic.py
```

在你的电脑具有摄像头的情况下，可以通过opencv来拍摄自己的照片，可能会需要一段时间，因为需要拍摄1w张呢（为了增加训练素材）

慢着，可不能直接用那个网站下好的图片集，我们需要将这个图片的格式规范化之后再进行训练，运行

```shell
python other_people_pic.py
```

可以将你下载的lfw图片集进行转化，转化为64×64的灰度图像供我们使用。

之后，对于我们的素材要进行划分，运行

```shell
python get_test_img.py
```

可以获得刚才获得的素材中根据二八原则分配的测试集和训练集。

## 开始训练

运行

```shell
python train_model.py
```

可以获得训练好的模型model.pkl

## 进行测试

运行

```shell
pyhton test_model.py
```

可以获得训练的结果，在我的测试中，准备率大约在98左右。

