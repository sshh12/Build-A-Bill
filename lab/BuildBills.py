
# coding: utf-8

# In[ ]:

import numpy as np
import os

from keras.models import load_model


# In[ ]:


def generate_bill():
    
    model = load_model(os.path.join('..', 'models', 'bill-gen.h5'))

    bill_fn = random.choice(os.listdir(os.path.join('..', 'data')))

    with open(os.path.join('..', 'data', bill_fn), 'r') as bill_file:
        rand_bill = bill_file.read()

    new_bill = rand_bill[:max_length]
    print("Starting w/:\n" + new_bill)

    for i in range(15000):

        X = np.zeros((1, max_length, num_chars))

        for t, char in enumerate(new_bill[-max_length:]):
            X[0, t, chr_to_int[char]] = 1

        preds = model.predict(X, verbose=0)[0]
        next_index = sample(preds, .6)

        if int_to_chr[next_index] != '*END*' or "body" == new_bill[-4:]:
            new_bill += int_to_chr[next_index]
        else:
            break

    with open('test.html', 'w') as bill_file:
        bill_file.write(new_bill)
        

