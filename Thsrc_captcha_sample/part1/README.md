# PART I: Use CNN model to test the captcha recognition.

VGG16 is currently used for testing, and other modules may be used for analysis in the future.

This module is slightly different from the actual VGG16.

## VGG16
Train Data: 8000<br>
Validation Data: 2000<br>
Test Data: 1000<br>

Epochs: 20, 25

## Model Code
```
tensor_in = Input((48, 140, 3))
tensor_out = tensor_in
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = BatchNormalization(axis=1)(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = BatchNormalization(axis=1)(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)

tensor_out = Flatten()(tensor_out)
tensor_out = Dropout(0.5)(tensor_out)

tensor_out = [Dense(19, name='digit1', activation='softmax')(tensor_out),\
            Dense(19, name='digit2', activation='softmax')(tensor_out),\
            Dense(19, name='digit3', activation='softmax')(tensor_out),\
            Dense(19, name='digit4', activation='softmax')(tensor_out)]

model = Model(inputs=tensor_in, outputs=tensor_out)
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.summary()
```

## VGG16 Model
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
batch_normalization (BatchNorma (None, 8, 31, 128)   32          conv2d_5[0][0]             
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 4, 15, 128)   0           batch_normalization[0][0]  
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 4, 15, 256)   295168      max_pooling2d_2[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 4, 15, 256)   590080      conv2d_6[0][0]             
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 2, 7, 256)    0           conv2d_7[0][0]             
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
digit1 (Dense)                  (None, 19)           29203       dropout[0][0]              
__________________________________________________________________________________________________
digit2 (Dense)                  (None, 19)           29203       dropout[0][0]              
__________________________________________________________________________________________________
digit3 (Dense)                  (None, 19)           29203       dropout[0][0]              
__________________________________________________________________________________________________
digit4 (Dense)                  (None, 19)           29203       dropout[0][0]              
==================================================================================================
Total params: 4,829,076
Trainable params: 4,829,056
Non-trainable params: 20
```
## VGG16 Model processing messages
```
1/160 [..............................] - ETA: 0s - loss: 21.3859 - digit1_loss: 5.7430 - digit2_loss: 5.5831 - digit3_loss: 5.2082 - digit4_loss: 4.8516 - digit1_accuracy: 0.0400 - digit2_accuracy: 0.0000e+00 - digit3_accuracy: 0.0800 - digit4_accuracy: 0.06002020-12-16 20:13:56.633325: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1479] CUPTI activity buffer flushed
2020-12-16 20:13:56.633868: I tensorflow/core/profiler/internal/gpu/device_tracer.cc:216]  GpuTracer has collected 454 callback api events and 454 activity events.
2020-12-16 20:13:56.672531: I tensorflow/core/profiler/rpc/client/save_profile.cc:168] Creating directory: logs/train/plugins/profile/2020_12_16_20_13_56
2020-12-16 20:13:56.702139: I tensorflow/core/profiler/rpc/client/save_profile.cc:174] Dumped gzipped tool data for trace.json.gz to logs/train/plugins/profile/2020_12_16_20_13_56/sxai1.trace.json.gz
2020-12-16 20:13:56.719447: I tensorflow/core/profiler/utils/event_span.cc:288] Generation of step-events took 0.172 ms

2020-12-16 20:13:56.725825: I tensorflow/python/profiler/internal/profiler_wrapper.cc:87] Creating directory: logs/train/plugins/profile/2020_12_16_20_13_56Dumped tool data for overview_page.pb to logs/train/plugins/profile/2020_12_16_20_13_56/sxai1.overview_page.pb
Dumped tool data for input_pipeline.pb to logs/train/plugins/profile/2020_12_16_20_13_56/sxai1.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to logs/train/plugins/profile/2020_12_16_20_13_56/sxai1.tensorflow_stats.pb
159/160 [============================>.] - ETA: 0s - loss: 15.0433 - digit1_loss: 3.7853 - digit2_loss: 3.7519 - digit3_loss: 3.7639 - digit4_loss: 3.7422 - digit1_accuracy: 0.0564 - digit2_accuracy: 0.0558 - digit3_accuracy: 0.0532 - digit4_accuracy: 0.05142020-12-16 20:14:00.169736: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 322560000 exceeds 10% of free system memory.
160/160 [==============================] - 5s 32ms/step - loss: 15.0214 - digit1_loss: 3.7800 - digit2_loss: 3.7464 - digit3_loss: 3.7582 - digit4_loss: 3.7368 - digit1_accuracy: 0.0570 - digit2_accuracy: 0.0561 - digit3_accuracy: 0.0534 - digit4_accuracy: 0.0518 - val_loss: 12.1169 - val_digit1_loss: 3.0055 - val_digit2_loss: 3.0372 - val_digit3_loss: 3.0301 - val_digit4_loss: 3.0442 - val_digit1_accuracy: 0.0665 - val_digit2_accuracy: 0.0565 - val_digit3_accuracy: 0.0480 - val_digit4_accuracy: 0.0555
Epoch 2/25
159/160 [============================>.] - ETA: 0s - loss: 10.1379 - digit1_loss: 2.4071 - digit2_loss: 2.4985 - digit3_loss: 2.5779 - digit4_loss: 2.6543 - digit1_accuracy: 0.2187 - d160/160 [==============================] - 4s 27ms/step - loss: 10.1234 - digit1_loss: 2.4030 - digit2_loss: 2.4948 - digit3_loss: 2.5742 - digit4_loss: 2.6514 - digit1_accuracy: 0.2200 - digit2_accuracy: 0.1963 - digit3_accuracy: 0.1696 - digit4_accuracy: 0.1439 - val_loss: 11.9096 - val_digit1_loss: 2.9526 - val_digit2_loss: 3.0164 - val_digit3_loss: 2.9768 - val_digit4_loss: 2.9637 - val_digit1_accuracy: 0.0490 - val_digit2_accuracy: 0.0565 - val_digit3_accuracy: 0.0480 - val_digit4_accuracy: 0.0620
Epoch 3/25
159/160 [============================>.] - ETA: 0s - loss: 4.8124 - digit1_loss: 0.8964 - digit2_loss: 1.2878 - digit3_loss: 1.3496 - digit4_loss: 1.2786 - digit1_accuracy: 0.6992 - di160/160 [==============================] - 4s 27ms/step - loss: 4.7952 - digit1_loss: 0.8929 - digit2_loss: 1.2827 - digit3_loss: 1.3450 - digit4_loss: 1.2745 - digit1_accuracy: 0.7004 - digit2_accuracy: 0.5736 - digit3_accuracy: 0.5545 - digit4_accuracy: 0.5767 - val_loss: 11.7330 - val_digit1_loss: 2.9525 - val_digit2_loss: 2.9685 - val_digit3_loss: 2.9137 - val_digit4_loss: 2.8985 - val_digit1_accuracy: 0.0490 - val_digit2_accuracy: 0.0565 - val_digit3_accuracy: 0.0485 - val_digit4_accuracy: 0.1410
Epoch 4/25
160/160 [==============================] - ETA: 0s - loss: 1.2849 - digit1_loss: 0.2756 - digit2_loss: 0.3895 - digit3_loss: 0.3535 - digit4_loss: 0.2664 - digit1_accuracy: 0.9234 - di160/160 [==============================] - 4s 27ms/step - loss: 1.2849 - digit1_loss: 0.2756 - digit2_loss: 0.3895 - digit3_loss: 0.3535 - digit4_loss: 0.2664 - digit1_accuracy: 0.9234 - digit2_accuracy: 0.8817 - digit3_accuracy: 0.8891 - digit4_accuracy: 0.9270 - val_loss: 9.3520 - val_digit1_loss: 2.3454 - val_digit2_loss: 2.3769 - val_digit3_loss: 2.3304 - val_digit4_loss: 2.2994 - val_digit1_accuracy: 0.1375 - val_digit2_accuracy: 0.2300 - val_digit3_accuracy: 0.4980 - val_digit4_accuracy: 0.5745
Epoch 5/25
159/160 [============================>.] - ETA: 0s - loss: 0.5268 - digit1_loss: 0.1394 - digit2_loss: 0.1502 - digit3_loss: 0.1350 - digit4_loss: 0.1022 - digit1_accuracy: 0.9650 - di160/160 [==============================] - 4s 27ms/step - loss: 0.5266 - digit1_loss: 0.1400 - digit2_loss: 0.1495 - digit3_loss: 0.1349 - digit4_loss: 0.1022 - digit1_accuracy: 0.9649 - digit2_accuracy: 0.9582 - digit3_accuracy: 0.9670 - digit4_accuracy: 0.9778 - val_loss: 4.2571 - val_digit1_loss: 1.0514 - val_digit2_loss: 1.0681 - val_digit3_loss: 1.0620 - val_digit4_loss: 1.0756 - val_digit1_accuracy: 0.9685 - val_digit2_accuracy: 0.9615 - val_digit3_accuracy: 0.9770 - val_digit4_accuracy: 0.9940
Epoch 6/25
158/160 [============================>.] - ETA: 0s - loss: 0.3207 - digit1_loss: 0.0897 - digit2_loss: 0.0917 - digit3_loss: 0.0801 - digit4_loss: 0.0591 - digit1_accuracy: 0.9773 - di160/160 [==============================] - 4s 27ms/step - loss: 0.3227 - digit1_loss: 0.0918 - digit2_loss: 0.0923 - digit3_loss: 0.0799 - digit4_loss: 0.0587 - digit1_accuracy: 0.9771 - digit2_accuracy: 0.9784 - digit3_accuracy: 0.9803 - digit4_accuracy: 0.9880 - val_loss: 0.7010 - val_digit1_loss: 0.1404 - val_digit2_loss: 0.1916 - val_digit3_loss: 0.1855 - val_digit4_loss: 0.1836 - val_digit1_accuracy: 0.9845 - val_digit2_accuracy: 0.9800 - val_digit3_accuracy: 0.9930 - val_digit4_accuracy: 0.9960
Epoch 7/25
158/160 [============================>.] - ETA: 0s - loss: 0.2177 - digit1_loss: 0.0647 - digit2_loss: 0.0594 - digit3_loss: 0.0518 - digit4_loss: 0.0418 - digit1_accuracy: 0.9838 - di160/160 [==============================] - 4s 27ms/step - loss: 0.2174 - digit1_loss: 0.0646 - digit2_loss: 0.0598 - digit3_loss: 0.0514 - digit4_loss: 0.0415 - digit1_accuracy: 0.9837 - digit2_accuracy: 0.9856 - digit3_accuracy: 0.9874 - digit4_accuracy: 0.9920 - val_loss: 0.1557 - val_digit1_loss: 0.0387 - val_digit2_loss: 0.0511 - val_digit3_loss: 0.0400 - val_digit4_loss: 0.0259 - val_digit1_accuracy: 0.9925 - val_digit2_accuracy: 0.9895 - val_digit3_accuracy: 0.9955 - val_digit4_accuracy: 0.9970
Epoch 8/25
159/160 [============================>.] - ETA: 0s - loss: 0.1562 - digit1_loss: 0.0505 - digit2_loss: 0.0421 - digit3_loss: 0.0348 - digit4_loss: 0.0287 - digit1_accuracy: 0.9857 - di160/160 [==============================] - 4s 27ms/step - loss: 0.1562 - digit1_loss: 0.0505 - digit2_loss: 0.0422 - digit3_loss: 0.0347 - digit4_loss: 0.0289 - digit1_accuracy: 0.9856 - digit2_accuracy: 0.9885 - digit3_accuracy: 0.9916 - digit4_accuracy: 0.9937 - val_loss: 0.1234 - val_digit1_loss: 0.0373 - val_digit2_loss: 0.0469 - val_digit3_loss: 0.0278 - val_digit4_loss: 0.0114 - val_digit1_accuracy: 0.9880 - val_digit2_accuracy: 0.9890 - val_digit3_accuracy: 0.9935 - val_digit4_accuracy: 0.9975
Epoch 9/25
160/160 [==============================] - ETA: 0s - loss: 0.1098 - digit1_loss: 0.0307 - digit2_loss: 0.0305 - digit3_loss: 0.0260 - digit4_loss: 0.0227 - digit1_accuracy: 0.9909 - di160/160 [==============================] - 4s 27ms/step - loss: 0.1098 - digit1_loss: 0.0307 - digit2_loss: 0.0305 - digit3_loss: 0.0260 - digit4_loss: 0.0227 - digit1_accuracy: 0.9909 - digit2_accuracy: 0.9921 - digit3_accuracy: 0.9933 - digit4_accuracy: 0.9951 - val_loss: 0.0944 - val_digit1_loss: 0.0261 - val_digit2_loss: 0.0388 - val_digit3_loss: 0.0232 - val_digit4_loss: 0.0062 - val_digit1_accuracy: 0.9930 - val_digit2_accuracy: 0.9885 - val_digit3_accuracy: 0.9935 - val_digit4_accuracy: 0.9990
Epoch 10/25
160/160 [==============================] - ETA: 0s - loss: 0.0825 - digit1_loss: 0.0266 - digit2_loss: 0.0221 - digit3_loss: 0.0185 - digit4_loss: 0.0153 - digit1_accuracy: 0.9914 - di160/160 [==============================] - 4s 27ms/step - loss: 0.0825 - digit1_loss: 0.0266 - digit2_loss: 0.0221 - digit3_loss: 0.0185 - digit4_loss: 0.0153 - digit1_accuracy: 0.9914 - digit2_accuracy: 0.9955 - digit3_accuracy: 0.9949 - digit4_accuracy: 0.9962 - val_loss: 0.0618 - val_digit1_loss: 0.0208 - val_digit2_loss: 0.0256 - val_digit3_loss: 0.0119 - val_digit4_loss: 0.0035 - val_digit1_accuracy: 0.9940 - val_digit2_accuracy: 0.9935 - val_digit3_accuracy: 0.9985 - val_digit4_accuracy: 0.9995
Epoch 11/25
158/160 [============================>.] - ETA: 0s - loss: 0.0756 - digit1_loss: 0.0214 - di160/160 [==============================] - 4s 25ms/step - loss: 0.0757 - digit1_loss: 0.0212 - digit2_loss: 0.0218 - digit3_loss: 0.0200 - digit4_loss: 0.0127 - digit1_accuracy: 0.9935 - digit2_accuracy: 0.9935 - digit3_accuracy: 0.9950 - digit4_accuracy: 0.9975 - val_loss: 0.0868 - val_digit1_loss: 0.0301 - val_digit2_loss: 0.0341 - val_digit3_loss: 0.0176 - val_digit4_loss: 0.0050 - val_digit1_accuracy: 0.9920 - val_digit2_accuracy: 0.9905 - val_digit3_accuracy: 0.9965 - val_digit4_accuracy: 0.9990
Epoch 12/25
160/160 [==============================] - ETA: 0s - loss: 0.0524 - digit1_loss: 0.0146 - di160/160 [==============================] - 4s 25ms/step - loss: 0.0524 - digit1_loss: 0.0146 - digit2_loss: 0.0134 - digit3_loss: 0.0135 - digit4_loss: 0.0109 - digit1_accuracy: 0.9955 - digit2_accuracy: 0.9960 - digit3_accuracy: 0.9960 - digit4_accuracy: 0.9976 - val_loss: 0.0820 - val_digit1_loss: 0.0270 - val_digit2_loss: 0.0370 - val_digit3_loss: 0.0147 - val_digit4_loss: 0.0034 - val_digit1_accuracy: 0.9925 - val_digit2_accuracy: 0.9900 - val_digit3_accuracy: 0.9970 - val_digit4_accuracy: 0.9990
Epoch 13/25
159/160 [============================>.] - ETA: 0s - loss: 0.0396 - digit1_loss: 0.0136 - di160/160 [==============================] - 4s 26ms/step - loss: 0.0396 - digit1_loss: 0.0135 - digit2_loss: 0.0100 - digit3_loss: 0.0081 - digit4_loss: 0.0080 - digit1_accuracy: 0.9959 - digit2_accuracy: 0.9971 - digit3_accuracy: 0.9979 - digit4_accuracy: 0.9980 - val_loss: 0.0771 - val_digit1_loss: 0.0250 - val_digit2_loss: 0.0301 - val_digit3_loss: 0.0155 - val_digit4_loss: 0.0066 - val_digit1_accuracy: 0.9945 - val_digit2_accuracy: 0.9925 - val_digit3_accuracy: 0.9950 - val_digit4_accuracy: 0.9980
Epoch 14/25
160/160 [==============================] - ETA: 0s - loss: 0.0621 - digit1_loss: 0.0174 - di160/160 [==============================] - 4s 25ms/step - loss: 0.0621 - digit1_loss: 0.0174 - digit2_loss: 0.0183 - digit3_loss: 0.0136 - digit4_loss: 0.0127 - digit1_accuracy: 0.9948 - digit2_accuracy: 0.9951 - digit3_accuracy: 0.9967 - digit4_accuracy: 0.9962 - val_loss: 0.0858 - val_digit1_loss: 0.0341 - val_digit2_loss: 0.0291 - val_digit3_loss: 0.0181 - val_digit4_loss: 0.0045 - val_digit1_accuracy: 0.9925 - val_digit2_accuracy: 0.9925 - val_digit3_accuracy: 0.9965 - val_digit4_accuracy: 0.9990
Epoch 15/25
160/160 [==============================] - ETA: 0s - loss: 0.0455 - digit1_loss: 0.0132 - di160/160 [==============================] - 4s 25ms/step - loss: 0.0455 - digit1_loss: 0.0132 - digit2_loss: 0.0138 - digit3_loss: 0.0108 - digit4_loss: 0.0077 - digit1_accuracy: 0.9966 - digit2_accuracy: 0.9956 - digit3_accuracy: 0.9973 - digit4_accuracy: 0.9987 - val_loss: 0.0767 - val_digit1_loss: 0.0411 - val_digit2_loss: 0.0198 - val_digit3_loss: 0.0130 - val_digit4_loss: 0.0027 - val_digit1_accuracy: 0.9870 - val_digit2_accuracy: 0.9930 - val_digit3_accuracy: 0.9965 - val_digit4_accuracy: 0.9985
Epoch 16/25
160/160 [==============================] - ETA: 0s - loss: 0.0330 - digit1_loss: 0.0110 - di160/160 [==============================] - 4s 26ms/step - loss: 0.0330 - digit1_loss: 0.0110 - digit2_loss: 0.0084 - digit3_loss: 0.0065 - digit4_loss: 0.0071 - digit1_accuracy: 0.9969 - digit2_accuracy: 0.9980 - digit3_accuracy: 0.9980 - digit4_accuracy: 0.9985 - val_loss: 0.0631 - val_digit1_loss: 0.0195 - val_digit2_loss: 0.0230 - val_digit3_loss: 0.0134 - val_digit4_loss: 0.0071 - val_digit1_accuracy: 0.9950 - val_digit2_accuracy: 0.9945 - val_digit3_accuracy: 0.9975 - val_digit4_accuracy: 0.9975
Epoch 17/25
158/160 [============================>.] - ETA: 0s - loss: 0.0288 - digit1_loss: 0.0099 - di160/160 [==============================] - 4s 26ms/step - loss: 0.0291 - digit1_loss: 0.0098 - digit2_loss: 0.0069 - digit3_loss: 0.0083 - digit4_loss: 0.0040 - digit1_accuracy: 0.9967 - digit2_accuracy: 0.9983 - digit3_accuracy: 0.9974 - digit4_accuracy: 0.9992 - val_loss: 0.0724 - val_digit1_loss: 0.0210 - val_digit2_loss: 0.0332 - val_digit3_loss: 0.0171 - val_digit4_loss: 0.0011 - val_digit1_accuracy: 0.9955 - val_digit2_accuracy: 0.9930 - val_digit3_accuracy: 0.9970 - val_digit4_accuracy: 0.9995
Epoch 18/25
160/160 [==============================] - ETA: 0s - loss: 0.0305 - digit1_loss: 0.0122 - di160/160 [==============================] - 4s 26ms/step - loss: 0.0305 - digit1_loss: 0.0122 - digit2_loss: 0.0087 - digit3_loss: 0.0059 - digit4_loss: 0.0037 - digit1_accuracy: 0.9965 - digit2_accuracy: 0.9976 - digit3_accuracy: 0.9989 - digit4_accuracy: 0.9995 - val_loss: 0.0734 - val_digit1_loss: 0.0208 - val_digit2_loss: 0.0307 - val_digit3_loss: 0.0205 - val_digit4_loss: 0.0014 - val_digit1_accuracy: 0.9965 - val_digit2_accuracy: 0.9925 - val_digit3_accuracy: 0.9970 - val_digit4_accuracy: 0.9995
Epoch 00018: early stopping
```
## Verify Testing Data
Total captchas: 1000<br>
Error: 7<br>
Success Rate: 0.993<br>

## Verify on website
Total captchas: 100<br>
Error: 5<br>
Success Rate: 0.95<br>

## Result Figures
###### digital 1
![alt text](https://github.com/reumng120/Learning_AI/blob/main/Thsrc_captcha_sample/part1/vgg16-cnn-25-dg1.png?raw=true)

###### digital 2
![alt text](https://github.com/reumng120/Learning_AI/blob/main/Thsrc_captcha_sample/part1/vgg16-cnn-25-dg2.png?raw=true)

###### digital 3
![alt text](https://github.com/reumng120/Learning_AI/blob/main/Thsrc_captcha_sample/part1/vgg16-cnn-25-dg3.png?raw=true)

###### digital 4
![alt text](https://github.com/reumng120/Learning_AI/blob/main/Thsrc_captcha_sample/part1/vgg16-cnn-25-dg4.png?raw=true)
