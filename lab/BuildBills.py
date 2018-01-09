
# coding: utf-8

# In[1]:

import numpy as np
import pickle
import random
import os

from keras.models import load_model


# In[2]:

# Options

max_length = 120


# In[3]:


def sample(preds, temperature):
    
    preds = np.asarray(preds).astype('float64')
    
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)


# In[4]:


def get_tables():
    
    with open(os.path.join('..', 'models', 'char-table.pickle'), 'rb') as table_file:
        chr_to_int = pickle.load(table_file)
        
    int_to_chr = { ix: char for char, ix in chr_to_int.items() }
    
    return chr_to_int, int_to_chr, len(chr_to_int)


# In[5]:


def generate_bill(save_fn, model_fn, sample_temp, chr_to_int, int_to_chr, seed=None, max_new_len=20000):
    
    model = load_model(os.path.join('..', 'models', model_fn))
    
    if not seed:

        bill_fn = random.choice(os.listdir(os.path.join('..', 'data')))

        with open(os.path.join('..', 'data', bill_fn), 'r') as bill_file:
            rand_bill = bill_file.read()

        seed = rand_bill
        
    new_bill = seed[:max_length]

    for i in range(max_new_len):

        X = np.zeros((1, max_length, num_chars))

        for t, char in enumerate(new_bill[-max_length:]):
            X[0, t, chr_to_int[char]] = 1

        preds = model.predict(X, verbose=0)[0]
        next_index = sample(preds, .6)

        if int_to_chr[next_index] != '*END*' or "body" == new_bill[-4:]:
            new_bill += int_to_chr[next_index]
        else:
            break

    with open(os.path.join('..', 'lab', 'tests', save_fn), 'w') as bill_file:
        bill_file.write(new_bill)
        


# In[ ]:


if __name__ == "__main__":
    
    chr_to_int, int_to_chr, num_chars = get_tables()


# In[ ]:


if __name__ == "__main__":
    
    seed1 = ("<body><p>S 73 IS</p><p/><center><p>112th CONGRESS</p></center>\n"
             "<p/><center><p>1st Session</p></center>\n"
             "<p/><center><p>S.")
            
    seed2 = ("<body><p>HR 3958 IH</p><p/><center><p>112th CONGRESS</p></center>\n"
             "<p/><center><p>2d Session</p></center>\n"
             "<p/><center><p>")
    
    seeds = [seed1, seed2]
    models = ['bill-gen.h5', 'bill-gen2.h5']
            
    for seed in seeds:
        
        for model_fn in models:
            
            for sample_temp in [.25, .5, 1, 2, 4]:
                
                test_name = "seed-{}-with-{}-at{}.html".format(seeds.index(seed), model_fn.replace('.', ''), sample_temp)
                
                print(test_name)
                
                generate_bill(test_name, model_fn, sample_temp, chr_to_int, int_to_chr, seed=seed)
                

