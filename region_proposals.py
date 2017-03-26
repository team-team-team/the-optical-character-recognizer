"""
This is a region proposal network for keras that will be used to detect words.

Copyright (C) 2017 Alec Graves

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

Contact me at shadysource2, gmail.com
"""

from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.models import Model

class region_proposer(object):
    'Region proposer object to predict bounding boxes around text'

    def __init__(self, scale=10/10):
        self.model = self.create_model()
        self.scale = scale

    def create_model(self):
        'Returns the build model'
        _input = Input((256,256,1))
        #x = Conv2D(int(32*self.scale), 3, 3, activation='relu')(_input)
        #x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Conv2D(int(64*self.scale), 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2,2))(x)
        x = Conv2D(int(128*self.scale), 3, 3, activation='relu')(x)
        x = Conv2D(int(64*self.scale), 1, 1, activation='relu')(x)
        x = Conv2D(int(128*self.scale), 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2,2))(x)
        x = Conv2D(int(256*self.scale), 3, 3, activation='relu')(x)
        x = Conv2D(int(128*self.scale), 1, 1, activation='relu')(x)
        mid = Conv2D(int(256*self.scale), 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2,2))(mid)
        x = Conv2D(int(512*self.scale), 3, 3, activation='relu')(x)
        x = Conv2D(int(256*self.scale), 1, 1, activation='relu')(x)
        x = Conv2D(int(512*self.scale), 3, 3, activation='relu')(x)
        x = Conv2D(int(256*self.scale), 1, 1, activation='relu')(x)
        x = Conv2D(int(512*self.scale), 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2,2))(x)
        x = Conv2D(int(1024*self.scale), 3, 3, activation='relu')(x)
        x = Conv2D(int(512*self.scale), 1, 1, activation='relu')(x)
        x = Conv2D(int(1024*self.scale), 3, 3, activation='relu')(x)
        x = Conv2D(int(512*self.scale), 1, 1, activation='relu')(x)
        x = Conv2D(int(1024*self.scale), 3, 3, activation='relu')(x)
        x = Conv2D(int(1024*self.scale), 3, 3, activation='relu')(x)
        x = Conv2D(int(1024*self.scale), 3, 3, activation='relu')(x)
        x = ZeroPadding2D([i - j for i, j in zip(mid.shape, x.shape)])
        out = Concatenate(axis=3)([x, mid])
        model = Model(inputs=_input, outputs=[out])
        return model

    def load_weights(self, path='region_proposer.h5'):
        'Load pretrained weights'
        self.model.load_weights(path)

    def train(self, path='region_proposer.h5'):
        'Train the model on generated data'
        pass