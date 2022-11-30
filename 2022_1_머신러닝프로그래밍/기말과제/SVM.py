#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import Image as Jupyter_Image


# In[2]:


import cv2
import os
import random


# In[3]:


IMAGE_SIZE = 200


# In[4]:


CATEGORIES = ['1', '2', '3', '4', '5']


# In[5]:


training_data = []
for category in CATEGORIES:
    category_path = os.path.join('.', 'train', category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(category_path):
        if '.png' in img:
            o_image = cv2.imread(os.path.join(category_path, img))
            new_image = cv2.resize(o_image, (IMAGE_SIZE, IMAGE_SIZE))
            training_data.append([os.path.join(category_path, img), new_image, class_num])


# In[6]:


testing_data = []
for category in CATEGORIES:
    category_path = os.path.join('.', 'test', category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(category_path):
        if '.png' in img:
            o_image = cv2.imread(os.path.join(category_path, img))
            new_image = cv2.resize(o_image, (IMAGE_SIZE, IMAGE_SIZE))
            testing_data.append([os.path.join(category_path, img), new_image, class_num])


# In[7]:


train_data=[]
train_label=[]

for o_image, categories, label in training_data:
    train_data.append(categories)
    train_label.append(label)

train_data=np.array(train_data).reshape(len(training_data), -1)
train_label=np.array(train_label)


# In[8]:


test_data=[]
test_label=[]

for o_image, categories, label in testing_data:
    test_data.append(categories)
    test_label.append(label)

test_data=np.array(test_data).reshape(len(test_data), -1)
test_label=np.array(test_label)


# In[9]:


plt.imshow(np.array(train_data[5]).reshape(200, -1))
print(CATEGORIES[train_label[5]])


# In[ ]:





# In[10]:


from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(train_data, train_label)


# In[11]:


predict_result = svc.predict(test_data)


# In[12]:


from sklearn.metrics import accuracy_score
print("Accuracy on unknown data is",accuracy_score(test_label,predict_result))


# In[13]:


from sklearn.metrics import classification_report
print("Accuracy on unknown data is",classification_report(test_label,predict_result))


# In[14]:


result = pd.DataFrame({'label' : test_label, 'predicted' : predict_result})


# In[15]:


result


# In[ ]:




