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

###### Webmail, social media, cloud storage를 비롯한 많은 종류의 온라인 서비스들은 abusing bot의 위협으로 부터 벗어나기 위해 CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart )를 defense mechanism으로 도입하기 시작하였습니다. 그러나, computer vision과 machine learning algorithm의 발전에 따라, CAPTCHA는 여전히 위 기술을 사용하는 abusing bot에 의해 파훼될 수 있습니다. 저희는 본 연구를 통해 CAPTCHA 중 하나의 scheme을 파훼하는 framework를 제시함으로서 간단한 computer vision skill과 machine learning algorithm으로도 CAPTCHA가 뚫릴 수 있음을 보여주고, 이를통해 text-based CAPTCHA defense mechanism이 취약함을 알리며 많은 온라인 서비스 보안의 안정성에 대해 물음표를 던질 것입니다.

# 2. Background

## CAPTCHA
###### CAPTCHA는 온라인 서비스의 보안을 위해 bot과 일반유저를 구분하는 test로 크게 다음과 같이 2가지로 분류 될 수 있습니다. 

* ###### Text-based CAPTCHAs : 유저가 noise가 끼어있고 distorted 되어있는 글자를 인식하고 타이핑하는 방식

* ###### Image-based CAPTCHAs : 유저가 CAPTCHA mechanism이 제시한 image set 중 request 받은 image를 고르는 방식


![]({{ site.url }}/img/BACKGROUND_CAPTCHA.PNG)


## Machine Learning Algorithms for Image detection
###### CAPTCHA defense mechanism을 파훼하는 데 사용되는 머신러닝 알고리즘은 image detection과 classificiation에 주로 이용되는 알고리즘으로 CNN (Convolutional Neural Network)가 주로 이용되고 있습니다.

* ###### Alexnet : Alex Krizhevsky가 Ilya Sutskever and Geoffrey Hinton와 함께 만든 CNN으로 2012년 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) top-5 error 15.3%를 달성하며 1위를 차지했습니다. 
![]({{ site.url }}/img/BACKGROUND_ALEXNET.PNG)


# 3. Datasets

###### 본 연구에 사용된 dataset 선정 기준은 다음과 같습니다.
* ###### 얼마나 흔하게 볼 수 있는가?
* ###### 얼마나 보편적인 CAPCTHA의 특징을 가지고 있는가?
* ###### 잘 정제된 traning image를 얻기 힘들다고 가정했을 때, traning image가 너무 많지는 않은가?

###### 위 기준에 따라 우리는  아래 사진과 같이 1) 회원가입시 가장 흔히 볼 수 있는 CAPTCHA이며, 2) noise line, blur 등 text-based CAPTCHA의 대표적인 성격을 잘 띄고 있고, 3) traning image 1000개, test image 50개의 작은 규모의 dataset을 고르게 되었습니다. (dataset from Wilhelmy, Rodrigo Rosas, Horacio) 

![]({{ site.url }}/img/CAPTCHA_BEFORE.PNG)

# 4. Proposed Framework

![]({{ site.url }}/img/PROPOSED_ALEX.png)


###### 위 제시된 그림은 본 연구에서 제시하는 framework의 overview이며 training session과 inference session으로 나뉘게 됩니다. 

* ###### Training Session : 1000장의 image sample을 각각 preprocessing 하여 모델의 정확도를 높입니다.

* ###### Inference session : 50장의 test sample들이 trained model에 입력되며 결과를 예측합니다. 모델의 정확도를 측정하기 위해 예측값과 실제값을 비교하는 추가 workload가 존재합니다.

## Preprocessing

* ###### Image thresholding : 주어진 task는 배경과 문자열이 확실하게 구분할 수 있으므로 0~255 값인 image intensity 중 절반 값인 127을 기준으로 높으면 흰색(255), 낮으면 검은색(0)으로 변환시키는 binary inversion을 수행합니다. 

~~~ ruby
X_train_pre=[]
for j in range(1000):
    ret,thresh1 = cv2.threshold(X_train[j],127,255,cv2.THRESH_BINARY_INV)
~~~



* ![]({{ site.url }}/img/PREPROSED_THRESHOLD.PNG)




* ###### Morphological opening : binary inversion 이후, 남아있는 noise line을 제거해주기 위해서 image속 작은 object를 지우기 위해 사용하는 방법인 opening을 이용합니다. opening은 주어진 커널을 통해 dilation 이후 erosion하여 main letter와 멀리 떨어져 있는 object를 지울 수 있습니다.

~~~ ruby
X_train_pre=[] #Preprocessed X_train set

for n in range(num_train_samples):

    ret,thresh1 = cv2.threshold(X_train[n],127,255,cv2.THRESH_BINARY_INV) #Otsu's segment
    kernel = np.ones((3,3),np.uint8) 
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel) # Morphological opening
    X_train_pre.append(opening)

plt.imshow(X_train_pre[0], cmap='gray')
~~~

* ![]({{ site.url }}/img/PREPROPOSED_MOR.PNG)

* ###### Horizontal cropping : 상대적으로 정확한 segmentation을 위해 image detection에 필요없는 여백부분을 제거합니다. 

~~~ ruby
# Horizontal cropping
intensity = 0 
intensity_array = []
len_column = len(X_train[0][0])
len_row = len(X_train[0])

for n in range(num_train_samples):
    for k in range(len_column):
        for m in range(len_row):
                intensity += X_train_pre[n][m,k]
        
        if(intensity > 255*15):
            intensity_array.append(k)
            
        intensity = 0
        
    col1 = min(intensity_array)
    col2 = max(intensity_array) + 5
   
    X_train_pre[n] = X_train_pre[n][:, col1:col2]
    del intensity_array[:]

plt.imshow(X_train_pre[0],cmap='gray')
~~~

* ![]({{ site.url }}/img/PREPROPOSED_HOR.PNG)

* ###### Segmentation : proposed model에 입력할 letter 단위 input을 만들기 위해 주어진 image를 1/5씩 자릅니다. horizontal cropping을 통해서 일괄적으로 1/5씩 자르더라도 resonable한 letter단위 input을 만들어낼 수 있습니다.


~~~ ruby
X_train_segment=[]

for n in range(num_train_samples):
    for num in range(5):
        x_leng = len(X_train_pre[n][0])
        x1 = (int)(num*x_leng/5)
        x2 = (int)((num+1)*x_leng/5)
        img = X_train_pre[n][:, x1:x2]
        X_train_segment.append(img)

f, axes = plt.subplots(nrows=1, ncols=5)
for k in range(5):
    axes[k].imshow(X_train_segment[k], cmap='gray')
    axes[k].set_yticks([])
plt.show()
~~~

* ![]({{ site.url }}/img/PREPROPOSED_SEG.PNG)

### Training

~~~ ruby
 def AlexNet(input_size, num_classes, summary=True):
    input_layer = Input(input_size)
 
  # Layer 1
    conv1 = Conv2D(96, (11, 11), padding='same', strides=4,
    kernel_regularizer=l2(1e-4))(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
 
  # Layer 2
    conv2 = Conv2D(256, (5, 5), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
 
  # Layer 3
    conv3 = Conv2D(384, (3, 3), padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
 
  # Layer 4
    conv4 = Conv2D(384, (3, 3), padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
 
  # Layer 5
    conv5 = Conv2D(256, (3, 3), padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = MaxPooling2D(pool_size=(2, 2))(conv5)
 
  # Layer 6
    conv6 = Flatten()(conv5)
    conv6 = Dense(4096)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.5)(conv6)
 
  # Layer 7
    conv7 = Dense(4096)(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.5)(conv7)
 
  # Layer 8
    conv8 = Dense(num_classes)(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('softmax')(conv8)
 
    output_layer = conv8
 
    model = keras.Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    if summary:
      model.summary()

   return model
 input_size = [64, 64, 1]
 num_classes = 36

 model = AlexNet(input_size, num_classes=num_classes)

 X_train, X_valid, Y_train, Y_valid = train_test_split(img_data, label_data, test_size=0.2)
 # creating the final model

 MODEL_SAVE_FOLDER_PATH = '/home/esoc/cwlee/AIX/alex_model'
 model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}.hdf5'
 checkpoint = ModelCheckpoint(filepath=model_path,
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=True)


 hist = model.fit(X_train, Y_train, batch_size=64, epochs=50, verbose=1, validation_data=(X_valid, Y_valid), callbacks=[checkpoint])
~~~

### Inference
~~~ ruby
model = AlexNet(input_size, num_classes=num_classes)
model.load_weights("alex_latest_model.hdf5")


for n in range(num_test_samples):
  for num in range(5):
    x_leng = len(X_test_pre[n][0])
    x1 = (int)(num*x_leng/5)
    x2 = (int)((num+1)*x_leng/5)
    img = X_test_pre[n][:, x1:x2]
    img = cv2.resize(img,(64, 64), interpolation=cv2.INTER_CUBIC)
    img = img[np.newaxis]
    img = np.expand_dims(np.array(img),3)/255
    y = model.predict(img)
    
    # mapping function matches the prediction value to the letters
    y_label = y_label + map_func(y)
  y_test_pred.append(y_label)
  y_label=""

 print(y_test_pred)

 for n in range(50):
   for m in range(5):
     if(y_test_pred[n][m]!=y_test[n][m]):
       count=count+1

 accuracy = 100 - (count/250 * 100)
 print("Test Accuracy = ",accuracy,"%")
~~~

# 5. Experiment

## Result

* ![]({{ site.url }}/img/RESULT_ALEX.PNG)

# 6. Conclusion

## Analysis
###### image thresholding, morphological opening, horizontal cropping, segmentation으로 이루어진 data preprocessing과 alexnet을 이용하여 총 250개의 letter 중에서 239개의 letter를 맞추며 95.6%의 정확도를 기록했습니다. 따라서 본 연구는 text-based CAPTCHA 중 하나를, 기본적인 computer vision techniques와 machine learning algorthims을 가지고 무력화 시킬 수 있음을 보여주었습니다. 

## Limitation and Future work
###### 비록 주어진 dataset이 많은 noise와 적은 training sample로 인해 text-based CAPTCHA중 파훼하기 어려운 편에 속하나 다른 sophisticated한 text-based CAPTCHA mechanism을 대상으로 하지는 않아, 모든 text-based CAPTCHA defense mechanism이 보안에 취약함을 증명하긴 힘듭니다. 하지만 본 연구의 실험결과가 보여주듯, CAPTCHA mechanism이 fix되있고 발전하지 않는다면, 충분히 해당 CAPTCHA에 optimized된 bot 공격에 취약할 것임을 밝힙니다. 추가연구를 계획한다면 최대한 본 연구와 비슷한 수준의 기술로, 좀 더 다양한 캡챠 이미지를 무력화 시킬 수 있다면, 본 연구가 제시하는 결론의 타당성을 높여줄 것이라 예상합니다.

## Related Work

###### CAPCHA defensce mechanism을 공격하는 수많은 연구들이 발표되었지만, 대부분의 연구는 주어진 CAPTCHA image를 본 연구에서와 비슷한 방식으로 preprocessing하고, optimized된 machine learning algorithms을 사용하여 공격의 정밀성을 높입니다. [1]-[7]

###### 다만 좀 더 다양한 set의 CAPTCHA image들과 다양한 algorithms을 적용시켜 CAPTCHA-solving-framework의 유연성을 높였습니다. 하지만 처음에 제시하다시피 본 연구의 목적은 simple한 algorithm과 skill을 가지고도 CAPTCHA를 무력화 시킬 수 있음을 보여주는 것이기 때문에 위에 제시된 논문들의 방법은 본 연구에 적합하지 않다고 생각했습니다.

##### 역할
* ###### 김승현 : 발표 및 영상촬영
* ###### 김준수 : 코딩 및 아이디어 제시
* ###### 이상호 : 홈페이지 디자인
* ###### 황은별 : 자료조사

##### Source Code

![https://github.com/Junsu-Kim-AIX/AIX-Project](https://github.com/Junsu-Kim-AIX/AIX-Project)


# Reference

* [1] University of Cambridge Computer Laboratory, 15 JJ Thomson Avenue, Cambridge CB3 0FD, UK
* [2] A00799283 Rodrigo Wilhelmy MIT A00792403 Horacio Rosas MIT
* [3] Zhao, Nathan, Yi Liu, and Yijun Jiang. "CAPTCHA Breaking with Deep Learning."
* [4] Huang, Shih-Yu, et al. "An efficient segmentation algorithm for CAPTCHAs with line cluttering and 
character warping." Multimedia Tools and Applications 48.2 (2010): 267-289
* [5] Stark, Fabian, et al. "Captcha recognition with active deep learning." GCPR Workshop on New Challenges in Neural Computation. Vol. 10. 2015
* [6] Baek, Youngmin, et al. "Character Region Awareness for Text Detection." arXiv preprint arXiv:1904.01941
* [7] Wang, Ye, and Mi Lu. "An optimized system to solve text-based CAPTCHA." arXiv preprint arXiv:1806.07202