
# coding: utf-8

# In[1]:

# Imports

import requests
import os


# In[2]:

# API

def download_bills(congress):
    """
    Downloads all bills found from GovTrack api (for given Congress) to data/
    """
    bill_list = requests.get("https://www.govtrack.us/api/v2/bill?congress={}".format(congress)).json()
    
    for bill in bill_list["objects"]:
        
        try:
            
            name = bill["display_number"]
            url = bill["text_info"]["html_file"]
            
            html_bill = requests.get("https://www.govtrack.us/" + url).text
            
            with open(os.path.join('..', 'data', name + '.bill'), 'w') as bill_file:
                bill_file.write(html_bill)
                
            print(bill["title"])
            
        except Exception as e:
            print(e)


# In[3]:


if __name__ == "__main__":

    download_bills(114)
    download_bills(113)
    download_bills(112)

