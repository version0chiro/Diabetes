import time
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import MaxPooling2D, AvgPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import applications
from keras.models import Sequential, Model, load_model
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.regularizers import l2
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from ann_visualizer.visualize import ann_viz
import tensorflow as tf

import os
import tensorflow as tf

from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('test/normal/'):
    f.extend(filenames)
    break

model = load_model('dibaties_right.h5')
classes = ['Control', 'DM']
runTotal = len(f)
time1 = time.time()
for i in f:
    cur_img = cv2.imread('dataset/test/DM/'+str(i))
    cur_img = cv2.resize(cur_img, (200, 200))
    cur_img = np.expand_dims(cur_img, axis=0)
    print(classes[np.argmax(model.predict(cur_img))])
    image = cv2.putText(cur_img, 'OpenCV', (50, 50), cv2.LINE_AA,
                        1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('test', cur_img)
    cv2.waitKey()

timetotal = time.time() - time1
fps = float(runTotal/timetotal)
print("FPS=%.2f, total frames = %.0f , time = %.4f seconds" %
      (fps, runTotal, timetotal))
