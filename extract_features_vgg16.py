from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import numpy as np
from glob import glob
import os
import cv2
from sklearn.cluster import KMeans
import pandas as pd

data_dir = '/content/drive/My Drive/Dataset/Vietnamese-handwritten-letters/sample/images/lost2'

def extract_vgg16():
  print('Start to read images')
  #load vgg16 model
  model = VGG16(weights='imagenet', include_top=False)
  model.summary()
  imgs_glob = glob(os.path.join(data_dir, "*.jpg"))
  images = []
  filenames = []
  for i in range(len(imgs_glob)):
    img_path = imgs_glob[i];
    filename = os.path.basename(img_path)
    print('Extracting vgg16 features from ' + filename + '..., Total processing ' + str(i*100.0/len(imgs_glob)) + '%')
    filenames.append(filename)
    #load image with size 224x224
    img = image.load_img(img_path, target_size=(224, 224))
    #convert to array
    img_data = image.img_to_array(img)
    #extract features with vgg16 model
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    #reshape to 1d-array
    images.append(vgg16_feature_np.flatten())
  print('Read images successfully')
  return filenames, images

filenames, images = extract_vgg16()
filenames = pd.Series(filenames)
images = np.array(images)
df = pd.DataFrame(images)
print(df)
#Save as csv training data for cluster later
filenames.to_csv('/content/drive/My Drive/Dataset/Vietnamese-handwritten-letters/sample/preprocessed_data.csv')
np.savetxt('/content/drive/My Drive/Dataset/Vietnamese-handwritten-letters/sample/preprocessed_data.csv', df, delimiter=';')
# print(images.shape) 
