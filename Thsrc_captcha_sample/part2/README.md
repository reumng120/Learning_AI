# PART II: Modify the image pre-processing and train model.

I spent a lot of time adjusting the pre-processing of the picture.

In order to train more deeper and stable modules, the settings are slightly different from the reference.

## Captcha pre-processing

- Denoise the captcha image
```
denoise_img = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)
```

- Convert image to gray
```
ret, gray = cv2.threshold(denoise_img, 127, 255, cv2.THRESH_BINARY_INV)
```

- Dilate and Erode image
```
dilation = cv2.dilate(gray, kernel, iterations = 1)
erosion = cv2.erode(img, kernel, iterations = 1)
```

- Find regression
```
img = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
img[15:, 2:self.WIDTH - 2] = 0
imagedata = np.where(img == 255)

X = np.array([imagedata[1]])
Y = self.HEIGHT - imagedata[0]

poly_reg = PolynomialFeatures(degree = 2)
X_ = poly_reg.fit_transform(X.T)
regr = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
regr.fit(X_, Y)
```

- Processing the curve line
```
X2 = np.array([[i for i in range(0, self.WIDTH)]])
poly_reg = PolynomialFeatures(degree = 2)
X2_ = poly_reg.fit_transform(X2.T)
offset_Top_H = 3
check_W = 4

newimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for ele in np.column_stack([regr.predict(X2_).round(0),X2[0],] ):
    pos = self.HEIGHT - int(ele[0])
    whiteCount = 0
    blackCount = 0
    for item in newimg[pos-offset_Top_H:pos+offset_Top_H,int(ele[1])]:
        if item == 255:
            whiteCount += 1
        else:
            blackCount += 1
    
    if whiteCount >= check_W:
        newimg[pos-offset_Top_H:pos+offset_Top_H,int(ele[1])] = 0
    else:
        newimg[pos-offset_Top_H:pos+offset_Top_H,int(ele[1])] = 255 - newimg[pos-offset_Top_H:pos+offset_Top_H,int(ele[1])]

```

## My CNN Model
```

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                    
==================================================================================================
input_1 (InputLayer)            [(None, 48, 140, 3)] 0                                           
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 48, 140, 32)  896         input_1[0][0]                   
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 46, 138, 32)  9248        conv2d[0][0]                    
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 23, 69, 32)   0           conv2d_1[0][0]                  
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 23, 69, 64)   18496       max_pooling2d[0][0]             
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 21, 67, 64)   36928       conv2d_2[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 10, 33, 64)   0           conv2d_3[0][0]                  
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 10, 33, 128)  73856       max_pooling2d_1[0][0]           
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 8, 31, 128)   147584      conv2d_4[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 4, 15, 128)   0           conv2d_5[0][0]                  
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 4, 15, 256)   295168      max_pooling2d_2[0][0]           
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 4, 15, 256)   590080      conv2d_6[0][0]                  
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 4, 15, 256)   16          conv2d_7[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 2, 7, 256)    0           batch_normalization[0][0]       
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 2, 7, 512)    1180160     max_pooling2d_3[0][0]           
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 2, 7, 512)    2359808     conv2d_8[0][0]                  
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 2, 7, 512)    8           conv2d_9[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 1, 3, 512)    0           batch_normalization_1[0][0]     
__________________________________________________________________________________________________
flatten (Flatten)               (None, 1536)         0           max_pooling2d_4[0][0]           
__________________________________________________________________________________________________
dropout (Dropout)               (None, 1536)         0           flatten[0][0]                   
__________________________________________________________________________________________________
digit1 (Dense)                  (None, 20)           30740       dropout[0][0]                   
__________________________________________________________________________________________________
digit2 (Dense)                  (None, 20)           30740       dropout[0][0]                   
__________________________________________________________________________________________________
digit3 (Dense)                  (None, 20)           30740       dropout[0][0]                   
__________________________________________________________________________________________________
digit4 (Dense)                  (None, 20)           30740       dropout[0][0]                   
==================================================================================================
Total params: 4,835,208
Trainable params: 4,835,196
Non-trainable params: 12
```
## My CNN Model processing messages
```
1/208 [..............................] - ETA: 0s - loss: 22.6791 - digit1_loss: 5.7708 - digit2_loss: 5.6762 - digit3_loss: 5.8494 - digit4_loss: 5.3826 - digit1_accuracy: 0.1000 - digit2_accuracy: 0.0600 - digit3_accuracy: 0.0600 - digit4_accuracy: 0.08002020-12-22 14:29:14.235788: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1479] CUPTI activity buffer flushed
2020-12-22 14:29:14.236022: I tensorflow/core/profiler/internal/gpu/device_tracer.cc:216]  GpuTracer has collected 453 callback api events and 453 activity events.
2020-12-22 14:29:14.258625: I tensorflow/core/profiler/rpc/client/save_profile.cc:168] Creating directory: logs/train/plugins/profile/2020_12_22_14_29_14
2020-12-22 14:29:14.267102: I tensorflow/core/profiler/rpc/client/save_profile.cc:174] Dumped gzipped tool data for trace.json.gz to logs/train/plugins/profile/2020_12_22_14_29_14/sxai1.trace.json.gz
2020-12-22 14:29:14.270689: I tensorflow/core/profiler/utils/event_span.cc:288] Generation of step-events took 0.083 ms

2020-12-22 14:29:14.272847: I tensorflow/python/profiler/internal/profiler_wrapper.cc:87] Creating directory: logs/train/plugins/profile/2020_12_22_14_29_14Dumped tool data for overview_page.pb to logs/train/plugins/profile/2020_12_22_14_29_14/sxai1.overview_page.pb
Dumped tool data for input_pipeline.pb to logs/train/plugins/profile/2020_12_22_14_29_14/sxai1.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to logs/train/plugins/profile/2020_12_22_14_29_14/sxai1.tensorflow_stats.pb
208/208 [==============================] - ETA: 0s - loss: 14.6509 - digit1_loss: 3.6741 - digit2_loss: 3.6283 - digit3_loss: 3.6706 - digit4_loss: 3.6779 - digit1_accuracy: 0.0514 - digit2_accuracy: 0.0589 - digit3_accuracy: 0.0539 - digit4_accuracy: 0.05772020-12-22 14:29:18.758103: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 419328000 exceeds 10% of free system memory.
208/208 [==============================] - 6s 31ms/step - loss: 14.6509 - digit1_loss: 3.6741 - digit2_loss: 3.6283 - digit3_loss: 3.6706 - digit4_loss: 3.6779 - digit1_accuracy: 0.0514 - digit2_accuracy: 0.0589 - digit3_accuracy: 0.0539 - digit4_accuracy: 0.0577 - val_loss: 12.0911 - val_digit1_loss: 3.0085 - val_digit2_loss: 3.0450 - val_digit3_loss: 3.0465 - val_digit4_loss: 2.9912 - val_digit1_accuracy: 0.0542 - val_digit2_accuracy: 0.0608 - val_digit3_accuracy: 0.0688 - val_digit4_accuracy: 0.0600
Epoch 2/8
206/208 [============================>.] - ETA: 0s - loss: 9.0671 - digit1_loss: 2.0634 - digit2_208/208 [==============================] - 5s 26ms/step - loss: 9.0391 - digit1_loss: 2.0528 - digit2_loss: 2.2745 - digit3_loss: 2.3171 - digit4_loss: 2.3947 - digit1_accuracy: 0.3226 - digit2_accuracy: 0.2561 - digit3_accuracy: 0.2501 - digit4_accuracy: 0.2302 - val_loss: 12.5687 - val_digit1_loss: 3.2597 - val_digit2_loss: 3.1463 - val_digit3_loss: 3.1242 - val_digit4_loss: 3.0384 - val_digit1_accuracy: 0.1038 - val_digit2_accuracy: 0.0712 - val_digit3_accuracy: 0.0377 - val_digit4_accuracy: 0.0588
Epoch 3/8
207/208 [============================>.] - ETA: 0s - loss: 3.1518 - digit1_loss: 0.5763 - digit2_loss: 0.8646 - digit3_loss: 0.9013 - digit4_loss: 0.8097 - digit1_accuracy: 0.8056 - digit2_accur208/208 [==============================] - 5s 26ms/step - loss: 3.1433 - digit1_loss: 0.5744 - digit2_loss: 0.8621 - digit3_loss: 0.8995 - digit4_loss: 0.8073 - digit1_accuracy: 0.8063 - digit2_accuracy: 0.7111 - digit3_accuracy: 0.7013 - digit4_accuracy: 0.7403 - val_loss: 8.4110 - val_digit1_loss: 2.0403 - val_digit2_loss: 2.2698 - val_digit3_loss: 2.1078 - val_digit4_loss: 1.9931 - val_digit1_accuracy: 0.3419 - val_digit2_accuracy: 0.2188 - val_digit3_accuracy: 0.2912 - val_digit4_accuracy: 0.4015
Epoch 4/8
206/208 [============================>.] - ETA: 0s - loss: 0.8195 - digit1_loss: 0.1877 - digit2_loss: 0.2498 - digit3_loss: 0.2205 - digit4_loss: 0.1615 - digit1_accuracy: 0.9483 - digit2_accur208/208 [==============================] - 6s 26ms/step - loss: 0.8173 - digit1_loss: 0.1873 - digit2_loss: 0.2492 - digit3_loss: 0.2195 - digit4_loss: 0.1612 - digit1_accuracy: 0.9485 - digit2_accuracy: 0.9283 - digit3_accuracy: 0.9371 - digit4_accuracy: 0.9588 - val_loss: 1.1585 - val_digit1_loss: 0.2697 - val_digit2_loss: 0.3703 - val_digit3_loss: 0.3390 - val_digit4_loss: 0.1795 - val_digit1_accuracy: 0.9496 - val_digit2_accuracy: 0.9431 - val_digit3_accuracy: 0.9531 - val_digit4_accuracy: 0.9915
Epoch 5/8
208/208 [==============================] - ETA: 0s - loss: 0.3882 - digit1_loss: 0.1119 - digit2_loss: 0.1138 - digit3_loss: 0.0928 - digit4_loss: 0.0696 - digit1_accuracy: 0.9736 - digit2_accur208/208 [==============================] - 5s 26ms/step - loss: 0.3882 - digit1_loss: 0.1119 - digit2_loss: 0.1138 - digit3_loss: 0.0928 - digit4_loss: 0.0696 - digit1_accuracy: 0.9736 - digit2_accuracy: 0.9720 - digit3_accuracy: 0.9792 - digit4_accuracy: 0.9860 - val_loss: 0.2707 - val_digit1_loss: 0.0500 - val_digit2_loss: 0.1072 - val_digit3_loss: 0.0718 - val_digit4_loss: 0.0417 - val_digit1_accuracy: 0.9869 - val_digit2_accuracy: 0.9658 - val_digit3_accuracy: 0.9881 - val_digit4_accuracy: 0.9950
Epoch 6/8
206/208 [============================>.] - ETA: 0s - loss: 0.2356 - digit1_loss: 0.0653 - digit2_loss: 0.0693 - digit3_loss: 0.0555 - digit4_loss: 0.0455 - digit1_accuracy: 0.9847 - digit2_accur208/208 [==============================] - 5s 26ms/step - loss: 0.2354 - digit1_loss: 0.0650 - digit2_loss: 0.0688 - digit3_loss: 0.0553 - digit4_loss: 0.0463 - digit1_accuracy: 0.9848 - digit2_accuracy: 0.9830 - digit3_accuracy: 0.9870 - digit4_accuracy: 0.9912 - val_loss: 0.0743 - val_digit1_loss: 0.0143 - val_digit2_loss: 0.0299 - val_digit3_loss: 0.0217 - val_digit4_loss: 0.0084 - val_digit1_accuracy: 0.9954 - val_digit2_accuracy: 0.9904 - val_digit3_accuracy: 0.9977 - val_digit4_accuracy: 0.9988
Epoch 7/8
208/208 [==============================] - ETA: 0s - loss: 0.1686 - digit1_loss: 0.0522 - digit2_208/208 [==============================] - 5s 26ms/step - loss: 0.1686 - digit1_loss: 0.0522 - digit2_loss: 0.0441 - digit3_loss: 0.0428 - digit4_loss: 0.0295 - digit1_accuracy: 0.9876 - digit2_accuracy: 0.9891 - digit3_accuracy: 0.9900 - digit4_accuracy: 0.9943 - val_loss: 0.0718 - val_digit1_loss: 0.0082 - val_digit2_loss: 0.0288 - val_digit3_loss: 0.0257 - val_digit4_loss: 0.0090 - val_digit1_accuracy: 0.9988 - val_digit2_accuracy: 0.9862 - val_digit3_accuracy: 0.9958 - val_digit4_accuracy: 0.9981
Epoch 8/8
206/208 [============================>.] - ETA: 0s - loss: 0.1238 - digit1_loss: 0.0346 - digit2_loss: 0.0340 - digit3_loss: 0.0286 - digit4_loss: 0.0266 - digit1_accuracy: 0.9906 - digit2_accur208/208 [==============================] - 5s 26ms/step - loss: 0.1244 - digit1_loss: 0.0352 - digit2_loss: 0.0343 - digit3_loss: 0.0284 - digit4_loss: 0.0265 - digit1_accuracy: 0.9903 - digit2_accuracy: 0.9916 - digit3_accuracy: 0.9930 - digit4_accuracy: 0.9941 - val_loss: 0.0452 - val_digit1_loss: 0.0082 - val_digit2_loss: 0.0159 - val_digit3_loss: 0.0187 - val_digit4_loss: 0.0024 - val_digit1_accuracy: 0.9988 - val_digit2_accuracy: 0.9931 - val_digit3_accuracy: 0.9977 - val_digit4_accuracy: 0.9992
```
## Verify Testing Data
Total captchas: 1000

Error: 6 (3 duplicate captchas)

Success Rate: 0.997

## Verify on website
Not yet

## Result Figures
###### digital 1
![alt text](https://github.com/reumng120/Learning_AI/blob/main/Thsrc_captcha_sample/part2/cnn-cus1-b50-8-1-dg1.png?raw=true)

###### digital 2
![alt text](https://github.com/reumng120/Learning_AI/blob/main/Thsrc_captcha_sample/part2/cnn-cus1-b50-8-1-dg2.png?raw=true)

###### digital 3
![alt text](https://github.com/reumng120/Learning_AI/blob/main/Thsrc_captcha_sample/part2/cnn-cus1-b50-8-1-dg3.png?raw=true)

###### digital 4
![alt text](https://github.com/reumng120/Learning_AI/blob/main/Thsrc_captcha_sample/part2/cnn-cus1-b50-8-1-dg4.png?raw=true)
