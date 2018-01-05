
# coding: utf-8

# In[18]:

import requests
import os


# In[19]:


def download_bills(congress):
    
    bill_list = requests.get("https://www.govtrack.us/api/v2/bill?congress={}".format(congress)).json()
    
    for bill in json["objects"]:
        
        print(bill["title"])
        
        try:
            
            name = bill["display_number"]
            url = bill["text_info"]["html_file"]
            
            html_bill = requests.get("https://www.govtrack.us/" + url).text
            
            with open(os.path.join('data', name + '.bill'), 'w') as bill:
                bill.write(html_bill)
            
        except:
            pass


# In[20]:


if __name__ == "__main__":

    download_bills(112)

