import option as opt
from keras import models
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import Flatten
from keras.layers import Dense
from keras import layers
op = opt.Options()


def make_ann_model() :
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(op.max_unique_word,)))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    return model 


def make_emd_model() :
    model = models.Sequential()
    model.add(Embedding(op.max_unique_word, op.embd_size , input_length=op.max_len))
    model.add(Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    return model 

 


 

