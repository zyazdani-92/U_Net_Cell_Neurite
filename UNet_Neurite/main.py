#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Change train path on trainGene to DHM/PDHM training dataset to train
U-DHM and U-PDHM, respectively.
Also, change validation path on valGene, accordingly.
Save model as hdf5 file, with approprite name on ModelCheckpoint code line.
"""
from model import *
from data_128 import *
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import adam_v2


NUM_EPOCHS=50
BATCH_SIZE = 2
NUM_TRAIN_IMAGES= 450 
NUM_VAL_IMAGES= 50
save_to_diri = "data/DHM/save_path" 
# data augmentation
data_gen_args = dict(rotation_range=0.01,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    shear_range= 0.0,
                    zoom_range=0.0,
                    horizontal_flip=True, 
                    rescale = 1./255,
                    fill_mode='nearest')
                    #preprocessing_function = add_noise

trainGene = trainGenerator(BATCH_SIZE,'data/DHM/train/128','DHM_train','Mask_train',data_gen_args,save_to_dir = save_to_diri)
valGene = valGenerator(BATCH_SIZE,'data/DHM/val/128','DHM_val','Mask_val',data_gen_args)

# Unet deep learning model
model=UNet(img_shape=(128,128,1), out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
           dropout=0.3, batchnorm=True, maxpool=True, upconv=True, residual=False)
model.compile(optimizer = adam_v2.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

# setup callback function 
early_stopping = EarlyStopping(patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
model_checkpoint = ModelCheckpoint('DHM_Neurite.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Train the model
H=model.fit_generator(trainGene,steps_per_epoch=NUM_TRAIN_IMAGES/BATCH_SIZE,epochs=NUM_EPOCHS,
                      callbacks=[ model_checkpoint,tensorboard_callback,early_stopping,reduce_lr],
                      validation_data=valGene,
                      validation_steps=NUM_VAL_IMAGES/BATCH_SIZE)



# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H.epoch, H.history["loss"], label="train_loss")
plt.plot(H.epoch, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("DHM_loss_curve.png")

