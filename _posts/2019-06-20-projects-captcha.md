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

Webmail, social media, cloud storage를 비롯한 많은 종류의 온라인 서비스들은 abusing bot의 위협으로 부터 벗어나기 위해 CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart )를 defense mechanism으로 도입하기 시작하였다. 그러나, computer vision과 machine learning algorithm의 발전에 따라, CAPTCHA는 여전히 위 기술을 사용하는 abusing bot에 의해 파훼될 수 있다. 우리는 본 연구에서 CAPTCHA 중 하나를 예시로 들며, 파훼하는 framework를 제시함으로서 간단한 computer vision skill과 machine learning algorithm으로도 CAPTCHA가 뚫릴 수 있음을 보여주고, 이를통해 CAPTCHA-based defense mechanism이 취약함을 알리며 많은 온라인 서비스 보안의 안정성에 대해 물음표를 던질 것이다.. 

# 2. Datasets

본 연구에 사용된 dataset 선정 기준은 다음과 같다.
* 얼마나 흔하게 볼 수 있는가?
* 얼마나 보편적인 CAPCTHA의 특징을 가지고 있는가?
* 잘 정제된 traning image를 얻기 힘들다고 가정했을 때, traning image가 너무 많지는 않은가?

위 기준에 따라 우리는  아래 사진과 같이 1) 회원가입시 가장 흔히 볼 수 있는 CAPTCHA이며, 2) noise line, blur 등 text-based CAPTCHA의 대표적인 성격을 잘 띄고 있고, 3) traning image 1000개, test image 50개의 작은 규모의 dataset을 고르게 되었다. (dataset from Wilhelmy, Rodrigo Rosas, Horacio) 
![]({{ site.url }}/img/CAPTCHA_BEFORE.PNG)

