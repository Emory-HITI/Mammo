#!/usr/bin/env python
# coding: utf-8

# # Image Type Derivation Notebook
# ### Jiwoong Jason Jeong

# In[1]:


# load necessary libraries
import os
import numpy as np
import pandas as pd


# In[2]:


# load paths and csv
base_path = '/home/jupyter-jjjeon3/data/mammo/png_BR0_samples/extracted-images/'
merged_50_2 = pd.read_csv('/data/mammo/png_BR0_samples/merged_50_2_full_anon.csv', low_memory=False)  # low_memory=False as these csv files are large


# In[3]:


# formuating the final function to add the image types and derived laterality or position

def derive_ImageTypes(test_df):
    """test_df should be the dataframe that contains the 'ImageLaterality', 'ViewPosition', 'SeriesDescription' columns"""
    # initializing empty lists:
    DeriveFlag = []  # ImageLaterality: 0 (not derived), 1 (derived), 2 (need to derive)
    ImageLateralityFinal = []  # the image laterality either copied or taken from series description
    FinalImageType = []  # final image type: 2D, 3D, cview, ROI_SSC, ROI_SS, Other (not any of the other ones)

    for index, row in test_df.iterrows():  # iterating over all rows
        try:
            desc = row['SeriesDescription'].split(' ')  # splitting up the series description for checking
        except:
            DeriveFlag.append(2)  # update derive flag = 0 
            ImageLateralityFinal.append('NaN')  # update final ImageLaterality
            FinalImageType.append('other')  # update imagetype
            continue
        if row['ImageLaterality'] in ['L', 'R']:  # if lat exists, it's either 2D or cview or ROI
            if 'C-View' in desc:  # if it has C-View it's cview
                DeriveFlag.append(0)  # update derive flag = 0 
                ImageLateralityFinal.append(row['ImageLaterality'])  # update final ImageLaterality
                FinalImageType.append('cview')  # update imagetype
            else:  # if it has lat and is not cview, it's 2d
                DeriveFlag.append(0)  # update derive flag = 0 
                ImageLateralityFinal.append(row['ImageLaterality'])  # update final ImageLaterality
                FinalImageType.append('2D')  # update imagetype
        else:  # if lat doesn't exist, it's either 3D, ROI
            if 'Tomosynthesis' in desc:  # if it has Tomosynthesis Reconstruction is 3D
                DeriveFlag.append(1)  # update derive flag = 1
                ImageLateralityFinal.append(desc[0])  # update final ImageLaterality
                FinalImageType.append('3D')  # update imagetype
            elif 'Capture' in desc:  # if it has Secondary Capture, it's SecurView Secondary Capture ROI
                DeriveFlag.append(2)  # update derive flag = 1
                ImageLateralityFinal.append('NaN')  # update final ImageLaterality
                FinalImageType.append('ROI_SSC')  # update imagetype
            elif 'Screen Save' in desc:
                DeriveFlag.append(2)  # update derive flag = 1
                ImageLateralityFinal.append('NaN')  # update final ImageLaterality
                FinalImageType.append('ROI_SS')  # update imagetype
            else:
                DeriveFlag.append(2)  # update derive flag = 1
                ImageLateralityFinal.append('NaN')  # update final ImageLaterality
                FinalImageType.append('other')  # update imagetype
    
    # adding the new, extracted columns
    test_df['DeriveFlag'] = DeriveFlag
    test_df['ImageLateralityFinal'] = ImageLateralityFinal
    test_df['FinalImageType'] = FinalImageType
    
    return test_df


# In[5]:


# Extracting the 'messy' Image Type information columns (not necessary but slightly faster - more significant as dataframes get much larger)
raw_imgType_df = merged_50_2[['ImageLaterality', 'ViewPosition', 'SeriesDescription']].copy()
print(raw_imgType_df.shape)
raw_imgType_df.head()


# In[6]:


# running the derivation
raw_imgType_df_derived = derive_ImageTypes(raw_imgType_df)
raw_imgType_df_derived.head()


# # Speed Test (Full dataframe vs Partial dataframe)

# In[7]:


import time


# In[8]:


# full dataframe
start = time.time()
merged_50_2_derived = derive_ImageTypes(merged_50_2)
end = time.time()
print(end - start)


# In[9]:


# partial dataframe
start = time.time()
raw_imgType_df_derived = derive_ImageTypes(raw_imgType_df)
end = time.time()
print(end - start)


# In[ ]:




