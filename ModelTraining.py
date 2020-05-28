#!/usr/bin/env python
# coding: utf-8

# # Using MobileNet for our FACE RECOGNIZER
# 
# ### Loading the MobileNet Model

# Freeze all layers except the top 4, as we'll only be training the top 4

# In[1]:


from keras.applications import MobileNet

# MobileNet was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224 

# Re-loads the MobileNet model without the top or FC layers
MobileNet = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers 
# Layers are set to trainable as True by default
for layer in MobileNet.layers:
    layer.trainable = False
    
# Let's print our layers 
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# ### Let's make a function that returns our FC Head

# In[2]:


def lw(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model


# ### Let's add our FC Head back onto MobileNet

# In[3]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# Set our class number to 3 (Young, Middle, Old)
num_classes = 211

FC_Head = lw(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())


# ### Loading our Coin Dataset 

# In[4]:


from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'path of your training data'
validation_data_dir = 'path of your validation data'

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
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


# ### Training out Model
# - Note we're using checkpointing and early stopping

# In[5]:


from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("Projectcoin1.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 6413
nb_validation_samples = 844
 
epochs = #If you require more than 70% accuracy you have to give more than 50epochs
batch_size = 16

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)


# ### Loading our classifer
# 

# In[6]:


from keras.models import load_model

classifier = load_model('Projectcoin1.h5')


# ### Testing our classifer on some test images

# In[12]:


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

coin_dict = {"[1]": "1 Cent,Australian dollar,australia",
    "[2]": "2 Cents,Australian dollar,australia",
    "[3]": "5 Cents,Australian dollar,australia",
    "[4]": "10 Cents,Australian dollar,australia",
    "[5]": "20 Cents,Australian dollar,australia",
    "[6]": "50 Cents,Australian dollar,australia",
    "[7]": "1 Dollar,Australian dollar,australia",
    "[8]": "2 Dollars,Australian dollar,australia",
    "[9]": "1 Centavo,Brazilian Real,brazil",
    "[10]": "5 Centavos,Brazilian Real,brazil",
    "[11]": "10 Centavos,Brazilian Real,brazil",
    "[12]": "25 Centavos,Brazilian Real,brazil",
    "[13]": "1 Real,Brazilian Real,brazil",
    "[14]": "1 Penny,British Pound,united_kingdom",
    "[15]": "2 Pence,British Pound,united_kingdom",
    "[16]": "5 Pence,British Pound,united_kingdom",
    "[17]": "10 Pence,British Pound,united_kingdom",
    "[18]": "20 Pence,British Pound,united_kingdom",
    "[19]": "50 Pence,British Pound,united_kingdom",
    "[20]": "1 Pound,British Pound,united_kingdom",
    "[21]": "2 Pounds,British Pound,united_kingdom",
    "[22]": "1 Cent,Canadian Dollar,canada",
    "[23]": "5 Cents,Canadian Dollar,canada",
    "[24]": "10 Cents,Canadian Dollar,canada",
    "[25]": "25 Cents,Canadian Dollar,canada",
    "[26]": "50 Cents,Canadian Dollar,canada",
    "[27]": "1 Dollar,Canadian Dollar,canada",
    "[28]": "2 Dollars,Canadian Dollar,canada",
    "[29]": "1 Peso,Chilean Peso,chile",
    "[30]": "5 Pesos,Chilean Peso,chile",
    "[31]": "10 Pesos,Chilean Peso,chile",
    "[32]": "50 Pesos,Chilean Peso,chile",
    "[33]": "100 Pesos,Chilean Peso,chile",
    "[34]": "500 Pesos,Chilean Peso,chile",
    "[35]": "1 Jiao,Chinese Yuan Renminbi,china",
    "[36]": "5 Jiao,Chinese Yuan Renminbi,china",
    "[37]": "1 Yuan,Chinese Yuan Renminbi,china",
    "[38]": "10 Hellers,Czech Koruna,czech_republic",
    "[39]": "20 Hellers,Czech Koruna,czech_republic",
    "[40]": "50 Hellers,Czech Koruna,czech_republic",
    "[41]": "1 Koruna,Czech Koruna,czech_republic",
    "[42]": "2 Koruny,Czech Koruna,czech_republic",
    "[43]": "5 Korun,Czech Koruna,czech_republic",
    "[44]": "10 Korun,Czech Koruna,czech_republic",
    "[45]": "20 Korun,Czech Koruna,czech_republic",
    "[46]": "50 Korun,Czech Koruna,czech_republic",
    "[47]": "25 Ore,Danish Krone,denmark",
    "[48]": "50 Ore,Danish Krone,denmark",
    "[49]": "1 Krone,Danish Krone,denmark",
    "[50]": "2 Kroner,Danish Krone,denmark",
    "[51]": "5 Kroner,Danish Krone,denmark",
    "[52]": "10 Kroner,Danish Krone,denmark",
    "[53]": "20 Kroner,Danish Krone,denmark",
    "[54]": "1 euro Cent,Euro,spain",
    "[55]": "2 euro Cent,Euro,spain",
    "[56]": "5 euro Cent,Euro,spain",
    "[57]": "10 euro Cent,Euro,spain",
    "[58]": "20 euro Cent,Euro,spain",
    "[59]": "50 euro Cent,Euro,spain",
    "[60]": "1 Euro,Euro,spain",
    "[61]": "2 Euro,Euro,spain",
    "[62]": "10 Cents,Hong Kong dollar,hong_kong",
    "[63]": "50 Cents,Hong Kong dollar,hong_kong",
    "[64]": "1 Dollar,Hong Kong dollar,hong_kong",
    "[65]": "2 Dollars,Hong Kong dollar,hong_kong",
    "[66]": "5 Dollars,Hong Kong dollar,hong_kong",
    "[67]": "1 Forint,Hungarian Forint,hungary",
    "[68]": "2 Forint,Hungarian Forint,hungary",
    "[69]": "5 Forint,Hungarian Forint,hungary",
    "[70]": "10 Forint,Hungarian Forint,hungary",
    "[71]": "20 Forint,Hungarian Forint,hungary",
    "[72]": "50 Forint,Hungarian Forint,hungary",
    "[73]": "100 Forint,Hungarian Forint,hungary",
    "[74]": "200 Forint,Hungarian Forint,hungary",
    "[75]": "25 Paise,Indian Rupee,india",
    "[76]": "50 Paise,Indian Rupee,india",
    "[77]": "1 Rupee,Indian Rupee,india",
    "[78]": "2 Rupees,Indian Rupee,india",
    "[79]": "5 Rupees,Indian Rupee,india",
    "[80]": "10 Rupees,Indian Rupee,india",
    "[81]": "50 Rupiah,Indonesian Rupiah,indonesia",
    "[82]": "100 Rupiah,Indonesian Rupiah,indonesia",
    "[83]": "200 Rupiah,Indonesian Rupiah,indonesia",
    "[84]": "500 Rupiah,Indonesian Rupiah,indonesia",
    "[85]": "1000 Rupiah,Indonesian Rupiah,indonesia",
    "[86]": "5 Agorot,Israeli New Shekel,israel",
    "[87]": "10 Agorot,Israeli New Shekel,israel",
    "[88]": "1 2 New Sheqel,Israeli New Shekel,israel",
    "[89]": "1 New Sheqel,Israeli New Shekel,israel",
    "[90]": "2 New Sheqalim,Israeli New Shekel,israel",
    "[91]": "5 New Sheqalim,Israeli New Shekel,israel",
    "[92]": "10 New Sheqalim,Israeli New Shekel,israel",
    "[93]": "1 Yen,Japanese Yen,japan",
    "[94]": "5 Yen,Japanese Yen,japan",
    "[95]": "10 Yen,Japanese Yen,japan",
    "[96]": "50 Yen,Japanese Yen,japan",
    "[97]": "100 Yen,Japanese Yen,japan",
    "[98]": "500 Yen,Japanese Yen,japan",
    "[99]": "1 Won,Korean Won,south_korea",
    "[100]": "5 Won,Korean Won,south_korea",
    "[101]": "10 Won,Korean Won,south_korea",
    "[102]": "50 Won,Korean Won,south_korea",
    "[103]": "100 Won,Korean Won,south_korea",
    "[104]": "500 Won,Korean Won,south_korea",
    "[105]": "1 Sen,Malaysian Ringgit,malaysia",
    "[106]": "5 Sen,Malaysian Ringgit,malaysia",
    "[107]": "10 Sen,Malaysian Ringgit,malaysia",
    "[108]": "20 Sen,Malaysian Ringgit,malaysia",
    "[109]": "50 Sen,Malaysian Ringgit,malaysia",
    "[110]": "5 Centavos,Mexican peso,mexico",
    "[111]": "10 Centavos,Mexican peso,mexico",
    "[112]": "20 Centavos,Mexican peso,mexico",
    "[113]": "50 Centavos,Mexican peso,mexico",
    "[114]": "1 Peso,Mexican peso,mexico",
    "[115]": "2 Pesos,Mexican peso,mexico",
    "[116]": "5 Pesos,Mexican peso,mexico",
    "[117]": "10 Pesos,Mexican peso,mexico",
    "[118]": "5 Cents,New Zealand dollar,new_zealand",
    "[119]": "10 Cents,New Zealand dollar,new_zealand",
    "[120]": "20 Cents,New Zealand dollar,new_zealand",
    "[121]": "50 Cents,New Zealand dollar,new_zealand",
    "[122]": "1 Dollar,New Zealand dollar,new_zealand",
    "[123]": "2 Dollars,New Zealand dollar,new_zealand",
    "[124]": "50 Ore,Norwegian Krone,norway",
    "[125]": "1 Krone,Norwegian Krone,norway",
    "[126]": "5 Kroner,Norwegian Krone,norway",
    "[127]": "10 Kroner,Norwegian Krone,norway",
    "[128]": "20 Kroner,Norwegian Krone,norway",
    "[129]": "1 Rupee,Pakistan Rupee,pakistan",
    "[130]": "2 Rupees,Pakistan Rupee,pakistan",
    "[131]": "5 Rupees,Pakistan Rupee,pakistan",
    "[132]": "10 Rupees,Pakistan Rupee,pakistan",
    "[133]": "1 Sentimo,Philipine peso,philippines",
    "[134]": "5 Sentimos,Philipine peso,philippines",
    "[135]": "10 Sentimos,Philipine peso,philippines",
    "[136]": "25 Sentimos,Philipine peso,philippines",
    "[137]": "1 Piso,Philipine peso,philippines",
    "[138]": "5 Piso,Philipine peso,philippines",
    "[139]": "10 Piso,Philipine peso,philippines",
    "[140]": "1 Grosz,Polish Zloty,poland",
    "[141]": "2 Grosze,Polish Zloty,poland",
    "[142]": "5 Groszy,Polish Zloty,poland",
    "[143]": "10 Groszy,Polish Zloty,poland",
    "[144]": "20 Groszy,Polish Zloty,poland",
    "[145]": "50 Groszy,Polish Zloty,poland",
    "[146]": "1 Zloty,Polish Zloty,poland",
    "[147]": "2 Zlote,Polish Zloty,poland",
    "[148]": "5 Zlotych,Polish Zloty,poland",
    "[149]": "1 Kopek,Russian Ruble,russia",
    "[150]": "5 Kopeks,Russian Ruble,russia",
    "[151]": "10 Kopeks,Russian Ruble,russia",
    "[152]": "50 Kopeks,Russian Ruble,russia",
    "[153]": "1 Ruble,Russian Ruble,russia",
    "[154]": "2 Rubles,Russian Ruble,russia",
    "[155]": "5 Rubles,Russian Ruble,russia",
    "[156]": "10 Rubles,Russian Ruble,russia",
    "[157]": "1 Cent,Singapore Dollar,singapore",
    "[158]": "5 Cents,Singapore Dollar,singapore",
    "[159]": "10 Cents,Singapore Dollar,singapore",
    "[160]": "20 Cents,Singapore Dollar,singapore",
    "[161]": "50 Cents,Singapore Dollar,singapore",
    "[162]": "1 Dollar,Singapore Dollar,singapore",
    "[163]": "5 Dollars,Singapore Dollar,singapore",
    "[164]": "5 Cents,South African Rand,south_africa",
    "[165]": "10 Cents,South African Rand,south_africa",
    "[166]": "20 Cents,South African Rand,south_africa",
    "[167]": "50 Cents,South African Rand,south_africa",
    "[168]": "1 Rand,South African Rand,south_africa",
    "[169]": "2 Rand,South African Rand,south_africa",
    "[170]": "5 Rand,South African Rand,south_africa",
    "[171]": "10 Ore,Swedish Krona,sweden",
    "[172]": "50 Ore,Swedish Krona,sweden",
    "[173]": "1 Krona,Swedish Krona,sweden",
    "[174]": "2 Kronor,Swedish Krona,sweden",
    "[175]": "5 Kronor,Swedish Krona,sweden",
    "[176]": "10 Kronor,Swedish Krona,sweden",
    "[177]": "1 Rappen,Swiss Franc,switzerland",
    "[178]": "5 Rappen,Swiss Franc,switzerland",
    "[179]": "10 Rappen,Swiss Franc,switzerland",
    "[180]": "20 Rappen,Swiss Franc,switzerland",
    "[181]": "1 2 Franc,Swiss Franc,switzerland",
    "[182]": "1 Franc,Swiss Franc,switzerland",
    "[183]": "2 Francs,Swiss Franc,switzerland",
    "[184]": "5 Francs,Swiss Franc,switzerland",
    "[185]": "1 2 Dollar,taiwan Dollar,taiwan",
    "[186]": "1 Dollar,taiwan Dollar,taiwan",
    "[187]": "5 Dollars,taiwan Dollar,taiwan",
    "[188]": "10 Dollars,taiwan Dollar,taiwan",
    "[189]": "20 Dollars,taiwan Dollar,taiwan",
    "[190]": "50 Dollars,taiwan Dollar,taiwan",
    "[191]": "1 Satang,Thai Baht,thailand",
    "[192]": "5 Satang,Thai Baht,thailand",
    "[193]": "10 Satang,Thai Baht,thailand",
    "[194]": "25 Satang,Thai Baht,thailand",
    "[195]": "50 Satang,Thai Baht,thailand",
    "[196]": "1 Baht,Thai Baht,thailand",
    "[197]": "2 Baht,Thai Baht,thailand",
    "[198]": "5 Baht,Thai Baht,thailand",
    "[199]": "10 Baht,Thai Baht,thailand",
    "[200]": "1 Kurus,Turkish Lira,turkey",
    "[201]": "5 Kurus,Turkish Lira,turkey",
    "[202]": "10 Kurus,Turkish Lira,turkey",
    "[203]": "25 Kurus,Turkish Lira,turkey",
    "[204]": "50 Kurus,Turkish Lira,turkey",
    "[205]": "1 Lira,Turkish Lira,turkey",
    "[206]": "1 Cent,US Dollar,usa",
    "[207]": "5 Cents,US Dollar,usa",
    "[208]": "1 Dime,US Dollar,usa",
    "[209]": "1 4 Dollar,US Dollar,usa",
    "[210]": "1 2 Dollar,US Dollar,usa",
    "[211]": "1 Dollar,US Dollar,usa"}

coin_dict_n = {
    "1": "1 Cent,Australian dollar,australia",
    "2": "2 Cents,Australian dollar,australia",
    "3": "5 Cents,Australian dollar,australia",
    "4": "10 Cents,Australian dollar,australia",
    "5": "20 Cents,Australian dollar,australia",
    "6": "50 Cents,Australian dollar,australia",
    "7": "1 Dollar,Australian dollar,australia",
    "8": "2 Dollars,Australian dollar,australia",
    "9": "1 Centavo,Brazilian Real,brazil",
    "10": "5 Centavos,Brazilian Real,brazil",
    "11": "10 Centavos,Brazilian Real,brazil",
    "12": "25 Centavos,Brazilian Real,brazil",
    "13": "1 Real,Brazilian Real,brazil",
    "14": "1 Penny,British Pound,united_kingdom",
    "15": "2 Pence,British Pound,united_kingdom",
    "16": "5 Pence,British Pound,united_kingdom",
    "17": "10 Pence,British Pound,united_kingdom",
    "18": "20 Pence,British Pound,united_kingdom",
    "19": "50 Pence,British Pound,united_kingdom",
    "20": "1 Pound,British Pound,united_kingdom",
    "21": "2 Pounds,British Pound,united_kingdom",
    "22": "1 Cent,Canadian Dollar,canada",
    "23": "5 Cents,Canadian Dollar,canada",
    "24": "10 Cents,Canadian Dollar,canada",
    "25": "25 Cents,Canadian Dollar,canada",
    "26": "50 Cents,Canadian Dollar,canada",
    "27": "1 Dollar,Canadian Dollar,canada",
    "28": "2 Dollars,Canadian Dollar,canada",
    "29": "1 Peso,Chilean Peso,chile",
    "30": "5 Pesos,Chilean Peso,chile",
    "31": "10 Pesos,Chilean Peso,chile",
    "32": "50 Pesos,Chilean Peso,chile",
    "33": "100 Pesos,Chilean Peso,chile",
    "34": "500 Pesos,Chilean Peso,chile",
    "35": "1 Jiao,Chinese Yuan Renminbi,china",
    "36": "5 Jiao,Chinese Yuan Renminbi,china",
    "37": "1 Yuan,Chinese Yuan Renminbi,china",
    "38": "10 Hellers,Czech Koruna,czech_republic",
    "39": "20 Hellers,Czech Koruna,czech_republic",
    "40": "50 Hellers,Czech Koruna,czech_republic",
    "41": "1 Koruna,Czech Koruna,czech_republic",
    "42": "2 Koruny,Czech Koruna,czech_republic",
    "43": "5 Korun,Czech Koruna,czech_republic",
    "44": "10 Korun,Czech Koruna,czech_republic",
    "45": "20 Korun,Czech Koruna,czech_republic",
    "46": "50 Korun,Czech Koruna,czech_republic",
    "47": "25 Ore,Danish Krone,denmark",
    "48": "50 Ore,Danish Krone,denmark",
    "49": "1 Krone,Danish Krone,denmark",
    "50": "2 Kroner,Danish Krone,denmark",
    "51": "5 Kroner,Danish Krone,denmark",
    "52": "10 Kroner,Danish Krone,denmark",
    "53": "20 Kroner,Danish Krone,denmark",
    "54": "1 euro Cent,Euro,spain",
    "55": "2 euro Cent,Euro,spain",
    "56": "5 euro Cent,Euro,spain",
    "57": "10 euro Cent,Euro,spain",
    "58": "20 euro Cent,Euro,spain",
    "59": "50 euro Cent,Euro,spain",
    "60": "1 Euro,Euro,spain",
    "61": "2 Euro,Euro,spain",
    "62": "10 Cents,Hong Kong dollar,hong_kong",
    "63": "50 Cents,Hong Kong dollar,hong_kong",
    "64": "1 Dollar,Hong Kong dollar,hong_kong",
    "65": "2 Dollars,Hong Kong dollar,hong_kong",
    "66": "5 Dollars,Hong Kong dollar,hong_kong",
    "67": "1 Forint,Hungarian Forint,hungary",
    "68": "2 Forint,Hungarian Forint,hungary",
    "69": "5 Forint,Hungarian Forint,hungary",
    "70": "10 Forint,Hungarian Forint,hungary",
    "71": "20 Forint,Hungarian Forint,hungary",
    "72": "50 Forint,Hungarian Forint,hungary",
    "73": "100 Forint,Hungarian Forint,hungary",
    "74": "200 Forint,Hungarian Forint,hungary",
    "75": "25 Paise,Indian Rupee,india",
    "76": "50 Paise,Indian Rupee,india",
    "77": "1 Rupee,Indian Rupee,india",
    "78": "2 Rupees,Indian Rupee,india",
    "79": "5 Rupees,Indian Rupee,india",
    "80": "10 Rupees,Indian Rupee,india",
    "81": "50 Rupiah,Indonesian Rupiah,indonesia",
    "82": "100 Rupiah,Indonesian Rupiah,indonesia",
    "83": "200 Rupiah,Indonesian Rupiah,indonesia",
    "84": "500 Rupiah,Indonesian Rupiah,indonesia",
    "85": "1000 Rupiah,Indonesian Rupiah,indonesia",
    "86": "5 Agorot,Israeli New Shekel,israel",
    "87": "10 Agorot,Israeli New Shekel,israel",
    "88": "1 2 New Sheqel,Israeli New Shekel,israel",
    "89": "1 New Sheqel,Israeli New Shekel,israel",
    "90": "2 New Sheqalim,Israeli New Shekel,israel",
    "91": "5 New Sheqalim,Israeli New Shekel,israel",
    "92": "10 New Sheqalim,Israeli New Shekel,israel",
    "93": "1 Yen,Japanese Yen,japan",
    "94": "5 Yen,Japanese Yen,japan",
    "95": "10 Yen,Japanese Yen,japan",
    "96": "50 Yen,Japanese Yen,japan",
    "97": "100 Yen,Japanese Yen,japan",
    "98": "500 Yen,Japanese Yen,japan",
    "99": "1 Won,Korean Won,south_korea",
    "100": "5 Won,Korean Won,south_korea",
    "101": "10 Won,Korean Won,south_korea",
    "102": "50 Won,Korean Won,south_korea",
    "103": "100 Won,Korean Won,south_korea",
    "104": "500 Won,Korean Won,south_korea",
    "105": "1 Sen,Malaysian Ringgit,malaysia",
    "106": "5 Sen,Malaysian Ringgit,malaysia",
    "107": "10 Sen,Malaysian Ringgit,malaysia",
    "108": "20 Sen,Malaysian Ringgit,malaysia",
    "109": "50 Sen,Malaysian Ringgit,malaysia",
    "110": "5 Centavos,Mexican peso,mexico",
    "111": "10 Centavos,Mexican peso,mexico",
    "112": "20 Centavos,Mexican peso,mexico",
    "113": "50 Centavos,Mexican peso,mexico",
    "114": "1 Peso,Mexican peso,mexico",
    "115": "2 Pesos,Mexican peso,mexico",
    "116": "5 Pesos,Mexican peso,mexico",
    "117": "10 Pesos,Mexican peso,mexico",
    "118": "5 Cents,New Zealand dollar,new_zealand",
    "119": "10 Cents,New Zealand dollar,new_zealand",
    "120": "20 Cents,New Zealand dollar,new_zealand",
    "121": "50 Cents,New Zealand dollar,new_zealand",
    "122": "1 Dollar,New Zealand dollar,new_zealand",
    "123": "2 Dollars,New Zealand dollar,new_zealand",
    "124": "50 Ore,Norwegian Krone,norway",
    "125": "1 Krone,Norwegian Krone,norway",
    "126": "5 Kroner,Norwegian Krone,norway",
    "127": "10 Kroner,Norwegian Krone,norway",
    "128": "20 Kroner,Norwegian Krone,norway",
    "129": "1 Rupee,Pakistan Rupee,pakistan",
    "130": "2 Rupees,Pakistan Rupee,pakistan",
    "131": "5 Rupees,Pakistan Rupee,pakistan",
    "132": "10 Rupees,Pakistan Rupee,pakistan",
    "133": "1 Sentimo,Philipine peso,philippines",
    "134": "5 Sentimos,Philipine peso,philippines",
    "135": "10 Sentimos,Philipine peso,philippines",
    "136": "25 Sentimos,Philipine peso,philippines",
    "137": "1 Piso,Philipine peso,philippines",
    "138": "5 Piso,Philipine peso,philippines",
    "139": "10 Piso,Philipine peso,philippines",
    "140": "1 Grosz,Polish Zloty,poland",
    "141": "2 Grosze,Polish Zloty,poland",
    "142": "5 Groszy,Polish Zloty,poland",
    "143": "10 Groszy,Polish Zloty,poland",
    "144": "20 Groszy,Polish Zloty,poland",
    "145": "50 Groszy,Polish Zloty,poland",
    "146": "1 Zloty,Polish Zloty,poland",
    "147": "2 Zlote,Polish Zloty,poland",
    "148": "5 Zlotych,Polish Zloty,poland",
    "149": "1 Kopek,Russian Ruble,russia",
    "150": "5 Kopeks,Russian Ruble,russia",
    "151": "10 Kopeks,Russian Ruble,russia",
    "152": "50 Kopeks,Russian Ruble,russia",
    "153": "1 Ruble,Russian Ruble,russia",
    "154": "2 Rubles,Russian Ruble,russia",
    "155": "5 Rubles,Russian Ruble,russia",
    "156": "10 Rubles,Russian Ruble,russia",
    "157": "1 Cent,Singapore Dollar,singapore",
    "158": "5 Cents,Singapore Dollar,singapore",
    "159": "10 Cents,Singapore Dollar,singapore",
    "160": "20 Cents,Singapore Dollar,singapore",
    "161": "50 Cents,Singapore Dollar,singapore",
    "162": "1 Dollar,Singapore Dollar,singapore",
    "163": "5 Dollars,Singapore Dollar,singapore",
    "164": "5 Cents,South African Rand,south_africa",
    "165": "10 Cents,South African Rand,south_africa",
    "166": "20 Cents,South African Rand,south_africa",
    "167": "50 Cents,South African Rand,south_africa",
    "168": "1 Rand,South African Rand,south_africa",
    "169": "2 Rand,South African Rand,south_africa",
    "170": "5 Rand,South African Rand,south_africa",
    "171": "10 Ore,Swedish Krona,sweden",
    "172": "50 Ore,Swedish Krona,sweden",
    "173": "1 Krona,Swedish Krona,sweden",
    "174": "2 Kronor,Swedish Krona,sweden",
    "175": "5 Kronor,Swedish Krona,sweden",
    "176": "10 Kronor,Swedish Krona,sweden",
    "177": "1 Rappen,Swiss Franc,switzerland",
    "178": "5 Rappen,Swiss Franc,switzerland",
    "179": "10 Rappen,Swiss Franc,switzerland",
    "180": "20 Rappen,Swiss Franc,switzerland",
    "181": "1 2 Franc,Swiss Franc,switzerland",
    "182": "1 Franc,Swiss Franc,switzerland",
    "183": "2 Francs,Swiss Franc,switzerland",
    "184": "5 Francs,Swiss Franc,switzerland",
    "185": "1 2 Dollar,taiwan Dollar,taiwan",
    "186": "1 Dollar,taiwan Dollar,taiwan",
    "187": "5 Dollars,taiwan Dollar,taiwan",
    "188": "10 Dollars,taiwan Dollar,taiwan",
    "189": "20 Dollars,taiwan Dollar,taiwan",
    "190": "50 Dollars,taiwan Dollar,taiwan",
    "191": "1 Satang,Thai Baht,thailand",
    "192": "5 Satang,Thai Baht,thailand",
    "193": "10 Satang,Thai Baht,thailand",
    "194": "25 Satang,Thai Baht,thailand",
    "195": "50 Satang,Thai Baht,thailand",
    "196": "1 Baht,Thai Baht,thailand",
    "197": "2 Baht,Thai Baht,thailand",
    "198": "5 Baht,Thai Baht,thailand",
    "199": "10 Baht,Thai Baht,thailand",
    "200": "1 Kurus,Turkish Lira,turkey",
    "201": "5 Kurus,Turkish Lira,turkey",
    "202": "10 Kurus,Turkish Lira,turkey",
    "203": "25 Kurus,Turkish Lira,turkey",
    "204": "50 Kurus,Turkish Lira,turkey",
    "205": "1 Lira,Turkish Lira,turkey",
    "206": "1 Cent,US Dollar,usa",
    "207": "5 Cents,US Dollar,usa",
    "208": "1 Dime,US Dollar,usa",
    "209": "1 4 Dollar,US Dollar,usa",
    "210": "1 2 Dollar,US Dollar,usa",
    "211": "1 Dollar,US Dollar,usa"
}

def draw_test(name, pred, im):
    coin = coin_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, coin, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + coin_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("path of your validation data")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()

