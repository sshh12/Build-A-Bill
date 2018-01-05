
# coding: utf-8

# In[17]:

# Imports
import numpy as np
import random
import os

from keras.layers import Dense, Activation, LSTM, TimeDistributed, Dropout
from keras.optimizers import RMSprop
from keras.models import Sequential

import matplotlib.pyplot as plt


# In[18]:

# Parameters

max_length = 120

epochs = 60
batch_size = 256


# In[19]:


def get_bills(nb_bills=12, mix=True):
    
    all_bills = os.listdir(os.path.join('..', 'data'))
    
    if mix:
    
        return random.sample(all_bills, nb_bills)
    
    else:
        
        return all_bills[:nb_bills]


# In[20]:

# Load Data

def load_data(bills, max_length=max_length, step=1):

    chars = set()

    sequences, next_chars = [], []

    for bill_fn in bills:

        with open(os.path.join('..', 'data', bill_fn), 'r') as bill_file:
            bill = bill_file.read()

        chars |= set(bill)

        for i in range(0, len(bill) - max_length, step):
            sequences.append(bill[i: i + max_length])
            next_chars.append(bill[i + max_length])

        sequences.append(bill[-max_length:])
        next_chars.append('*END*')

    chars = sorted(list(chars)) + ['*END*']
    chr_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_chr = dict((i, c) for i, c in enumerate(chars))
    
    return (sequences, next_chars), (len(chars), chr_to_int, int_to_chr)


# In[21]:

# Process Data

def process_data(sequences, next_chars, num_chars, chr_to_int):

    X = np.zeros((len(sequences), max_length, num_chars), dtype=np.bool)
    y = np.zeros((len(sequences), num_chars), dtype=np.bool)
    
    for i, sequence in enumerate(sequences):
        
        for j, char in enumerate(sequence):
            
            X[i, j, chr_to_int[char]] = 1
            
        y[i, chr_to_int[next_chars[i]]] = 1
        
    return X, y


# In[22]:

# Get Model

def get_model(num_chars, max_length=max_length):

    model = Sequential()

    model.add(LSTM(512, input_shape=(max_length, num_chars), return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.5))

    model.add(Dense(num_chars))

    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.002))
    
    return model
    


# In[23]:


def sample(preds, temperature=1.0):
    
    preds = np.asarray(preds).astype('float64')
    
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)


# In[26]:

# (Run) Load Data

if __name__ == "__main__":
    
    bills = get_bills()
    
    (sequences, next_chars), (num_chars, chr_to_int, int_to_chr) = load_data(bills)
    
    X, y = process_data(sequences, next_chars, num_chars, chr_to_int)
    
    print(X.shape, y.shape)


# In[8]:

# (Run) Train

if __name__ == "__main__":
    
    model = get_model(num_chars)
    
    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=.1,
                        verbose=1,
                        callbacks=[])
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['TrainLoss', 'TestLoss'])
    plt.show()


# In[11]:


def generate_bill():

    bill_fn = random.choice(os.path.join('..', 'data'))

    with open(os.path.join('data', bill_fn), 'r') as bill_file:
        rand_bill = bill_file.read()

    new_bill = rand_bill[:max_length]
    print("Starting w/:\n" + new_bill)

    for i in range(15000):

        X = np.zeros((1, max_length, num_chars))

        for t, char in enumerate(new_bill[-max_length:]):
            X[0, t, chr_to_int[char]] = 1.

        preds = model.predict(X, verbose=0)[0]
        next_index = sample(preds, .9)

        if int_to_chr[next_index] != '*END*' or "body" == new_bill[-4:]:
            new_bill += int_to_chr[next_index]
        else:
            break

    with open('test.html', 'w') as bill_file:
        bill_file.write(new_bill)



# In[12]:


if __name__ == "__main__":
    
    generate_bill()

