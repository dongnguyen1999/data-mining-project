from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import numpy as np
from glob import glob
import os
import cv2
from sklearn.cluster import KMeans
import pandas as pd

# preprocessing with vgg16

data_dir = '/content/drive/My Drive/Dataset/Vietnamese-handwritten-letters/sample/images/raw2'


def cluster(image_source, training_data, k):
  print('Started training kmean model with k = ' + str(k))
  nb_clusters = k
  kmeans = KMeans(n_clusters = nb_clusters)
  y_mean = kmeans.fit_predict(training_data)
  print('Train model successfully')

  print('Started save clusters')
  parent_dir = "/content/drive/My Drive/Dataset/Vietnamese-handwritten-letters/sample/images/clusteredv2.3_"+str(k)
  os.mkdir(parent_dir)
  for i in range(nb_clusters):
    directory = str(i)
    save_dir = os.path.join(parent_dir, directory)
    os.mkdir(save_dir)
    cl = image_source[y_mean == i]
    for index, row in cl.iterrows():
      filename = row['filename']
      image = row['image']
      cv2.imwrite(os.path.join(save_dir, filename), image)
    print('Saved ' + str(i+1) +' clusters')


def read_imgs():
  print('Start to read raw images')
  imgs_glob = glob(os.path.join(data_dir, "*.jpg"))
  images = []
  filenames = []
  for i in range(len(imgs_glob)):
    img_path = imgs_glob[i];
    filename = os.path.basename(img_path)
    print('Loading image ' + filename + '..., Total processing ' + str(i*100.0/len(imgs_glob)) + '%')
    filenames.append(filename)
    images.append(cv2.imread(img_path))
  print('Read raw images successfully')
  return filenames, images

#load source images
filenames, images = read_imgs()
source = pd.concat([pd.Series(filenames), pd.Series(images)], axis=1, keys=['filename', 'image'])

#read processed data preprocessing with vgg16
data = pd.read_csv('/content/drive/My Drive/Dataset/Vietnamese-handwritten-letters/sample/preprocessed_data.csv', delimiter=';', header=None)

posible_k = [90, 94, 100, 120, 150, 180, 190]

for k in posible_k:
  cluster(source, data, k)
 
