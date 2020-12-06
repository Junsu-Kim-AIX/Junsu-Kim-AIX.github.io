---
layout: post
title: "CAPTCHA PROJECT : Building a Machine Reads Distorted Text"
categories: [projects]
comments: true

---


<!--more-->
# Author

* ###### 김준수  / 융합전자공학부 /  j0807s@hanyang.ac.kr
* ###### 김승현  / 융합전자공학부 /  seankim95@naver.com
* ###### 이상호  / 경제금융학부  /   ta4tsg@gmail.com
* ###### 황은별  / 무용학부      /   sstar0219@naver.com

# 1. Introduction

Webmail, social media, cloud storage를 비롯한 많은 종류의 온라인 서비스들은 abusing bot의 위협으로 부터 벗어나기 위해 CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart )를 defense mechanism으로 도입하기 시작하였습니다. 그러나, computer vision과 machine learning algorithm의 발전에 따라, CAPTCHA는 여전히 위 기술을 사용하는 abusing bot에 의해 파훼될 수 있습니다. 저희는 본 연구를 통해 CAPTCHA 중 하나의 scheme을 파훼하는 framework를 제시함으로서 간단한 computer vision skill과 machine learning algorithm으로도 CAPTCHA가 뚫릴 수 있음을 보여주고, 이를통해 text-based CAPTCHA defense mechanism이 취약함을 알리며 많은 온라인 서비스 보안의 안정성에 대해 물음표를 던질 것입니다.

# 2. Background

### CAPTCHA
CAPTCHA는 온라인 서비스의 보안을 위해 bot과 일반유저를 구분하는 test로 크게 다음과 같이 2가지로 분류 될 수 있습니다. 

* Text-based CAPTCHAs : 유저가 noise가 끼어있고 distorted 되어있는 글자를 인식하고 타이핑하는 방식

* Image-based CAPTCHAs : 유저가 CAPTCHA mechanism이 제시한 image set 중 request 받은 image를 고르는 방식


![]({{ site.url }}/img/BACKGROUND_CAPTCHA.PNG)


### Machine Learning Algorithms for Image detection
CAPTCHA defense mechanism을 파훼하는 데 사용되는 머신러닝 알고리즘은 image detection과 classificiation에 주로 이용되는 알고리즘으로 CNN (Convolutional Neural Network)가 주로 이용되고 있습니다.

* Alexnet : Alex Krizhevsky가 Ilya Sutskever and Geoffrey Hinton와 함께 만든 CNN으로 2012년 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) top-5 error 15.3%를 달성하며 1위를 차지했습니다. 
![]({{ site.url }}/img/BACKGROUND_ALEXNET.PNG)

* Vgg 19 : Oxford Robotics Institute의 Karen Simonyan과 Andrew Zisserman이 제시한 모델로, 2014년 ILSVRC에서 top-5 error 92.7% 를 달성하며 GoogLeNet (93.3%)에 이어 2번째를 차지했습니다.  
![]({{ site.url }}/img/BACKGROUND_VGG19.PNG)


# 3. Datasets

본 연구에 사용된 dataset 선정 기준은 다음과 같습니다.
* 얼마나 흔하게 볼 수 있는가?
* 얼마나 보편적인 CAPCTHA의 특징을 가지고 있는가?
* 잘 정제된 traning image를 얻기 힘들다고 가정했을 때, traning image가 너무 많지는 않은가?

위 기준에 따라 우리는  아래 사진과 같이 1) 회원가입시 가장 흔히 볼 수 있는 CAPTCHA이며, 2) noise line, blur 등 text-based CAPTCHA의 대표적인 성격을 잘 띄고 있고, 3) traning image 1000개, test image 50개의 작은 규모의 dataset을 고르게 되었습니다. (dataset from Wilhelmy, Rodrigo Rosas, Horacio) 
![]({{ site.url }}/img/CAPTCHA_BEFORE.PNG)

# 3. Proposed Framework
![]({{ site.url }}/img/PROPOSED_ARCH.png)


위 제시된 그림은 본 연구에서 제시하는 framework의 overview이며 training session과 inference session으로 나뉘게 됩니다. 또한 회색으로 칠해진 부분은 본 연구의 가장 challenging한 problem을 해결하는 핵심 아이디어임을 표시했습니다.

* Training Session : 1000장의 image sample을 각각 preprocessing 하여 모델의 정확도를 높이고, 특히 data augmentation approach로 부족한 sample의 양을 실제 데이터의 추가수집 없이 보충합니다. preprocessing으로 정제된 image는 transfer learning approach로 알고리즘의 맨 마지막 fully connected layer와 dense layer만을 학습하게 됩니다.

* Inference session : 50장의 test sample들이 trained model에 입력되며 결과를 예측합니다. 모델의 정확도를 측정하기 위해 예측값과 실제값을 비교하는 추가 workload가 존재합니다.

### Challenging Problem 1 : Noise Reduction

* Image thresholding : 주어진 task는 배경과 문자열이 확실하게 구분할 수 있으므로 0~255 값인 image intensity 중 절반 값인 127을 기준으로 높으면 흰색(255), 낮으면 검은색(0)으로 변환시키는 binary inversion을 수행합니다. 

~~~ ruby
X_train_pre=[]
for j in range(1000):
    ret,thresh1 = cv2.threshold(X_train[j],127,255,cv2.THRESH_BINARY_INV)
~~~

* Morphological opening

* Horizontal cropping

* Segmentation

### Challenging Problem 2 : Small Training Set

* Data augmentation

* Transfer learning

# 4. Experiment

### Comparison Group

### Result

# 5. Conclusion
기존 방식의 CAPTCHA는 흔하게 사용하는 방식으론 small idea인 data augmentation approach와 transfer learning을 사용했을때도 정확도가 ~이상나오므로 sophsticated bot에 대해 충분히 취약함을 알 수 있었다.

역할
* 김승현 : 발표 및 영상촬영
* 김준수 : 아이디어 제시 및 코딩
* 이상호 : 홈페이지 디자인
* 황은별 : 자료조사

# Reference