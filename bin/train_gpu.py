import pandas as pd
import numpy as np
import seaborn as sns
import math
import os
import cv2
import efficientnet.tfkeras as efn
from tqdm import tqdm
from google.colab.patches import cv2_imshow
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Input, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import ResNet50, ResNet50V2, DenseNet121, DenseNet201, Xception, InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_spli

PROJECT_DIR = os.path.join('drive', 'My Drive', 'kaggle_plant_pathology')
IMG_DIR = os.path.join(PROJECT_DIR, 'src', 'modified')
MODIFIED_IMG_DIR = os.path.join(PROJECT_DIR, 'src', 'modified')
DATA_OUTPUT = os.path.join(PROJECT_DIR, 'data')
MODEL_OUTPUT = os.path.join(PROJECT_DIR, 'model')
SUBMIT_OUTPUT =  os.path.join(PROJECT_DIR, 'submit')


def read_data(train=True, save=False):
  x = []
  y = []
  if train:
    label_df = pd.read_csv(os.path.join(PROJECT_DIR, 'src', 'train.csv'))
    for img_id in tqdm(label_df['image_id']):
      img_data = cv2.imread(os.path.join(IMG_DIR, '{}.jpg'.format(img_id)))
      img_data = cv2.resize(img_data, (512, 512))
      label_data = label_df[label_df.image_id == img_id].iloc[:, 1:5].values
      x.append(img_data)
      y.extend(label_data)
    x = {'train_x.npy': np.stack(x)}
    y = {'train_y.npy': np.stack(y)}
  else:
    label_df = pd.read_csv(os.path.join(PROJECT_DIR, 'src', 'test.csv'))
    for img_id in tqdm(label_df['image_id']):
      img_data = cv2.imread(os.path.join(IMG_DIR, '{}.jpg'.format(img_id)))
      img_data = cv2.resize(img_data, (512, 512))
      x.append(img_data)
    x = {'test_x.npy': np.stack(x)}
    y = {'test_id.npy': label_df['image_id'].values}
  if save:
    os.makedirs(DATA_OUTPUT, exist_ok=True)
    np.save(os.path.join(DATA_OUTPUT, list(x.keys())[0]), list(x.values())[0])
    np.save(os.path.join(DATA_OUTPUT, list(y.keys())[0]), list(y.values())[0])

  return list(x.values())[0], list(y.values())[0]

x, y = read_data(save=True)
test_x, test_id = read_data(train=False, save=True)


def modified_raw_data(train=True):
  os.makedirs(DATA_OUTPUT, exist_ok=True)
  if train:
    label_df = pd.read_csv(os.path.join(PROJECT_DIR, 'src', 'train.csv'))
    output_filename = 'train_df'
  else:
    label_df = pd.read_csv(os.path.join(PROJECT_DIR, 'src', 'test.csv'))
    output_filename = 'test_df'
  img_df = []
  for img_id in tqdm(label_df['image_id']):
    img_data = cv2.imread(os.path.join(IMG_DIR, '{}.jpg'.format(img_id)))
    img_data = cv2.resize(img_data, (380, 380))
    img_df.append(img_data.reshape(-1))
  img_df = np.stack(img_df)
  img_df = pd.DataFrame(img_df)
  img_df = pd.concat([label_df, img_df], axis=1)
  img_df.to_pickle(os.path.join(DATA_OUTPUT, output_filename))

modified_raw_data(train=True)
modified_raw_data(train=False)

def read_modified_data(train=True, save=False):
  x = []
  y = []
  if train:
    label_df = pd.read_csv(os.path.join(PROJECT_DIR, 'src', 'train.csv'))
    for img_id in tqdm(label_df['image_id']):
      img_data = cv2.imread(os.path.join(MODIFIED_IMG_DIR, 'modified_{}.jpg'.format(img_id)))
      #img_data = cv2.resize(img_data, resolution)
      label_data = label_df[label_df.image_id == img_id].iloc[:, 1:5].values
      x.append(img_data)
      y.extend(label_data)
    x = {'grabCut_528_train_x.npy': np.stack(x)}
    y = {'train_y.npy': np.stack(y)}
  else:
    label_df = pd.read_csv(os.path.join(PROJECT_DIR, 'src', 'test.csv'))
    for img_id in tqdm(label_df['image_id']):
      img_data = cv2.imread(os.path.join(MODIFIED_IMG_DIR, 'modified_{}.jpg'.format(img_id)))
      #img_data = cv2.resize(img_data, resolution)
      x.append(img_data)
    x = {'grabCut_528_test_x.npy': np.stack(x)}
    y = {'test_id.npy': label_df['image_id'].values}
  if save:
    os.makedirs(DATA_OUTPUT, exist_ok=True)
    np.save(os.path.join(DATA_OUTPUT, list(x.keys())[0]), list(x.values())[0])
    np.save(os.path.join(DATA_OUTPUT, list(y.keys())[0]), list(y.values())[0])

  return list(x.values())[0], list(y.values())[0]


def img_preprocessing(file_name):
  img_data = np.load(os.path.join(DATA_OUTPUT, file_name))
  modified_img = []
  resolution = 380
  i = 0
  error = []
  for img in tqdm(img_data):
    ori_img = img.copy()
    ori_denoise = cv2.fastNlMeansDenoising(ori_img, h=5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    denoise = cv2.fastNlMeansDenoising(gray, h=5)
    filter_status = False
    base = [denoise, blur, gray]
    for canny_base in base:
      canny = cv2.Canny(canny_base, 100, 200)
      x, y, w, h = cv2.boundingRect(canny)
      if w >= resolution//2 and h >= resolution//2:
        filter_status = True
        break
    if not filter_status:
      img = ori_denoise
      error.append(i)
    else:
      img = ori_denoise[y:y+h, x:x+w]
    img = cv2.resize(img, (resolution, resolution))
    modified_img.append(img)
    i += 1
  modified_img = np.stack(modified_img)
  np.save(os.path.join(DATA_OUTPUT, 'modified_denoise_r{}_'.format(resolution) + file_name), modified_img)
  print(error)
  return modified_img



def rotate(img, deg):
  for _ in range(deg//90):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
  return img


def image_augment(x, y):
  new_x = []
  new_y = []
  for label, img in tqdm(zip(y, x)):
    transformed_img = [rotate(img, 90), rotate(img, 270), cv2.blur(img, (5, 5))]
    for img_id in range(len(transformed_img)):
      new_y.append(label)
    new_x.extend(transformed_img)
  return np.stack(new_x), np.stack(new_y)


def res_block(inputs, filter_nums, strides=1):
  conv_1 = Conv2D(filter_nums, (3, 3), strides=strides,
                  padding='same')  # , kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
  bn_1 = BatchNormalization()
  activation = ReLU()
  # dropout = Dropout(rate=0.1)
  conv_2 = Conv2D(filter_nums, (3, 3), strides=1,
                  padding='same')  # , kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
  bn_2 = BatchNormalization()
  output = conv_1(inputs)
  output = bn_1(output, training=True)
  output = activation(output)
  # output = dropout(output, training=True)
  output = conv_2(output)
  output = bn_2(output, training=True)
  if strides != 1:
    short_path = Conv2D(filter_nums, (1, 1), strides=strides)
    identity = short_path(inputs)
  else:
    identity = inputs
  output = tf.keras.layers.add([output, identity])
  output = tf.nn.relu(output)
  return output


def ResNet(layers_dim, input_shape, model_name='resnet'):
  inputs = tf.keras.Input(shape=input_shape)
  x = Conv2D(64, (3, 3), strides=1)(inputs)
  x = BatchNormalization()(x, training=True)
  x = ReLU()(x)
  x = MaxPooling2D(pool_size=2, strides=1)(x)
  for _ in range(layers_dim[0]):
    x = res_block(x, 64)
    x = res_block(x, 64)
  for _ in range(layers_dim[1]):
    x = res_block(x, 128, 2)
    x = res_block(x, 128)
  for _ in range(layers_dim[2]):
    x = res_block(x, 256, 2)
    x = res_block(x, 256)
  for _ in range(layers_dim[3]):
    x = res_block(x, 512, 2)
    x = res_block(x, 512)
  x = GlobalAveragePooling2D()(x)
  output = Dense(4, activation='softmax')(x)
  model = tf.keras.Model(inputs=inputs, outputs=output, name=model_name)
  # tf.keras.utils.plot_model(model, to_file=os.path.join('drive', 'My Drive', 'kaggle_bengaliai', 'model_{}'.format(model_name), '{}-model.png'.format(model_name)))
  return model


def train_valid_split(data, portion=0.95):
  portion = int(portion * data.shape[0])
  return data[:portion], data[portion:]


def create_model(model_name, input_shape):
  if model_name == 'efn_b4':
    model = efn.EfficientNetB4(weights=None, classes=4)
  elif model_name == 'efn_b4_p':
    model = tf.keras.models.Sequential()
    model.add(efn.EfficientNetB4(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'efn_b5_p':
    model = tf.keras.models.Sequential()
    model.add(efn.EfficientNetB5(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'efn_b7_p':
    model = tf.keras.models.Sequential()
    model.add(efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'resnet18':
    model = ResNet([2, 2, 2, 2], input_shape=input_shape)
  elif model_name == 'densenet121_p':
    model = tf.keras.models.Sequential()
    model.add(DenseNet121(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'densenet201_p':
    model = tf.keras.models.Sequential()
    model.add(DenseNet201(input_shape=input_shape, weights='imagenet', include_top=False))

  if model_name.split('_')[-1] == 'p':
    model.add(GlobalAveragePooling2D())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))
  model.summary()
  return model


def lr_scheduler_config(epoch_num):
  if epoch_num < 15:
    lr = 0.001 * (3 ** int((-epoch_num) / 3))
  else:
    lr = 0.001 * (3 ** int(-15 / 3)) * (2 ** int((-epoch_num + 15) / 3))
  lr = max(0.000001, lr)
  print('[Learning Rate]: {}'.format(lr))
  return lr


shuffle_idx = np.random.permutation(x.shape[0])
x = x[shuffle_idx]
y = y[shuffle_idx]
train_x, valid_x = train_valid_split(x)
train_y, valid_y = train_valid_split(y)

train_data_gen_config = ImageDataGenerator(
  rotation_range=10,
  horizontal_flip=True,
  vertical_flip=True,
  # width_shift_range=0.1,
  # height_shift_range=0.1,
  # zoom_range=1,
  rescale=1. / 255,
  fill_mode='nearest',
  brightness_range=[0.5, 1.5],
  shear_range=0.1
)

train_data_gen_config.fit(train_x)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler_config)

batch_size = 8
model = create_model('efn_b7_p', (224, 224, 3))
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['categorical_accuracy'])
model.fit(
  train_data_gen_config.flow(train_x, train_y, batch_size=batch_size),
  validation_data=(valid_x.astype(np.float16) / 255, valid_y.astype(np.float16) / 255),
  steps_per_epoch=train_x.shape[0] / batch_size,
  epochs=25,
  callbacks=[lr_scheduler])


def train_valid_split(data, portion=0.95):
  portion = int(portion * data.shape[0])
  return data[:portion], data[portion:]


def lr_scheduler_config(epoch_num):
  if epoch_num < 15:
    lr = 0.001 * (3 ** int(-epoch_num / 5))
  else:
    lr = 0.001 * (3 ** int(-15 / 5)) * (2 ** int((-epoch_num + 15) / 3))
  print('[Learning Rate]: {}'.format(lr))
  return lr


def create_model(model_name):
  if model_name == 'efn_b4':
    model = efn.EfficientNetB4(weights=None, classes=4)
  elif model_name == 'efn_b4_p':
    model = tf.keras.models.Sequential()
    model.add(efn.EfficientNetB4(input_shape=(380, 380, 3), weights='imagenet', include_top=False))
  elif model_name == 'efn_b5_p':
    model = tf.keras.models.Sequential()
    model.add(efn.EfficientNetB5(input_shape=(456, 456, 3), weights='imagenet', include_top=False))
  elif model_name == 'resnet18':
    model = ResNet([2, 2, 2, 2], input_shape=(224, 224, 3))
  elif model_name == 'densenet121_p':
    model = tf.keras.models.Sequential()
    model.add(DenseNet121(input_shape=(224, 224, 3), weights='imagenet', include_top=False))
  elif model_name == 'densenet201_p':
    model = tf.keras.models.Sequential()
    model.add(DenseNet201(input_shape=(224, 224, 3), weights='imagenet', include_top=False))

  if model_name.split('_')[-1] == 'p':
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))
  model.summary()
  return model


def train(x, y, model=None, lr_rate=0.001, epochs=10, batch_size=64):
  train_x, valid_x = train_valid_split(x)
  train_y, valid_y = train_valid_split(y)
  shuffle_train_idx = np.random.permutation(train_x.shape[0])
  shuffle_valid_idx = np.random.permutation(valid_x.shape[0])
  train_x = train_x[shuffle_train_idx]
  train_y = train_y[shuffle_train_idx]
  valid_x = valid_x[shuffle_valid_idx]
  valid_y = valid_y[shuffle_valid_idx]
  lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler_config)
  if model is None:
    # with strategy.scope():
    model = create_model('efn_b4_p')
    model.compile(optimizer=Adam(learning_rate=lr_rate), loss=CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['categorical_accuracy'])
  else:
    # with strategy.scope():
    model.compile(optimizer=Adam(learning_rate=lr_rate), loss=CategoricalCrossentropy(),
                  metrics=['categorical_accuracy'])
  model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size,
            callbacks=[lr_scheduler])
  return model


x = resize_and_bgr2rgb(x)
x, y = image_augment(x, y)
trained_model = train(x, y, lr_rate=0.001, epochs=30, batch_size=8
