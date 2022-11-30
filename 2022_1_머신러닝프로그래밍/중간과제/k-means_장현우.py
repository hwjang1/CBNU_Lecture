#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random as rd
import numpy as np
from tqdm import tqdm


# In[2]:


K = 2


# In[3]:


print('-------- samples init --------')
Samples = []
Samples.append([0.0, 0.5])
Samples.append([1.0, 0.0])
Samples.append([0.0, 1.0])
Samples.append([1.0, 1.0])
Samples.append([2.0, 1.0])
Samples.append([1.0, 2.0])
Samples.append([2.0, 2.0])
Samples.append([3.0, 2.0])
Samples.append([6.0, 6.0])
Samples.append([6.0, 7.0])
Samples.append([7.0, 6.0])
Samples.append([7.0, 7.0])
Samples.append([7.0, 8.0])
Samples.append([8.0, 6.0])
Samples.append([8.0, 7.0])
Samples.append([8.0, 8.0])
Samples.append([8.0, 9.0])
Samples.append([9.0, 7.0])
Samples.append([9.0, 8.0])
Samples.append([9.0, 9.0])
print(Samples)


# In[4]:


df = pd.DataFrame(Samples, columns=['x', 'y'])
print('-------- to dataframe --------')
print(df.head())


# In[5]:


print('-------- centroid init --------')
centroids = []
for _ in range(K):
    #d = df.iloc[rd.randint(0, len(df) - 1)]
    d = df.iloc[_]
    centroids.append(np.array((d.x, d.y)))
print(centroids)


# In[6]:


print('-------- calc centroid euclidean distance --------')
for i, centroid in enumerate(centroids):
    df['centroid%s' % i] = df.apply(lambda x: np.linalg.norm(np.array((x['x'], x['y'])) - centroid), axis=1)
print(df)


# In[7]:


print('-------- "select_cluster" function init --------')
def select_cluster(rows):
    centroid_val = []
    for key in rows.keys():
        if 'centroid' in key:
            centroid_val.append({'key': key, 'val': rows[key]})
    return min(centroid_val, key=lambda item:item['val'])['key']


# In[8]:


print('-------- select cluster --------')
df['cluster'] = df.apply(lambda x: select_cluster(x), axis=1)
print(df)


# In[9]:


register_centroid = []


# In[10]:


for loop_count in range(10):
    print('-------- loop centroid init --------')
    loop_centroids = []
    for i in range(K):
        loop_data = df[df['cluster'] == 'centroid%s' % i]
        loop_x = loop_data['x'].sum() / len(loop_data)
        loop_y = loop_data['y'].sum() / len(loop_data)
        loop_centroids.append(np.array((loop_x, loop_y)))
    if not np.array_equal(register_centroid, loop_centroids):
        register_centroid = loop_centroids
    else:
        break
    print('-------- loop calc centroid euclidean distance --------')
    for i, centroid in enumerate(loop_centroids):
        df['centroid%s' % i] = df.apply(lambda x: np.linalg.norm(np.array((x['x'], x['y'])) - centroid), axis=1)
    print('-------- loop select cluster --------')
    df['cluster'] = df.apply(lambda x: select_cluster(x), axis=1)


# In[17]:


print('update centroid count : %s' % loop_count)


# In[22]:


for i in range(K):
    print('centroid_%s value -> %s, %s' % (i, register_centroid[i][0], register_centroid[i][1]))
    print('centroid_%s members -> start')
    print(df[df['cluster'] == 'centroid%s' % i])
    print('centroid_%s members -> end')
    print('\n')


# In[ ]:





# In[ ]:





# In[ ]:




