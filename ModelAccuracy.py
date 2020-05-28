#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model

classifier = load_model('Projectcoin1.h5')


# In[2]:


from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'C:/Users/DS/Desktop/ML-Ops/coins/data/train/'
validation_data_dir = 'C:/Users/DS/Desktop/ML-Ops/coins/data/validation/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')


# In[3]:


num_test = len(validation_generator.filenames[10])


# In[4]:


score = classifier.evaluate_generator(validation_generator, steps=num_test//1, verbose=1)


# In[6]:


accuracy = score[1]*100 # Converting it to percentage


# In[11]:


f= open("projectacc.txt","w+")
f.write(str(accuracy))
f.close()


# In[ ]:




