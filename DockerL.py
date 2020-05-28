#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
def check_if_string_in_file(file_name, string_to_search):
    with open(file_name, 'r') as read_obj:
        for line in read_obj:
            if string_to_search in line:
                return True
    return False

# Searching For CNN 
if check_if_string_in_file('ModelTraining.py', 'Conv2D'):
    os.system('echo "Yes, string found in file" ')
    os.system('sudo docker -divt /root/AutoAIProject1:/root/ --name Python_OS cnn_djd')

# Searching for Linear Regression
if check_if_string_in_file('ModelTraining.py', 'LinearRegression'):
    os.system('echo "Yes, string found in file" ')
    os.system('sudo docker -divt /root/AutoAIProject1:/root/ --name Python_OS linear_djd')

else:
    os.system('echo "Yes, string found in file" ')
    os.system('sudo docker -divt /root/AutoAIProject1:/root/ --name Python_OS basic_djd')


# In[ ]:




