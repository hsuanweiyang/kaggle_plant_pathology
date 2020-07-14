import pandas as pd
import numpy as np
import os
import cv2
import efficientnet.tfkeras as efn
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Input, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.applications import ResNet50, ResNet50V2, DenseNet121, DenseNet201, Xception, InceptionResNetV2
from kaggle_datasets import KaggleDatasets


AUTO = tf.data.experimental.AUTOTUNE
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)

GCS_DS_PATH = KaggleDatasets().get_gcs_path()

PROJECT_DIR = os.path.join('..', 'input', 'plant-pathology-2020-fgvc7')
IMG_DIR = os.path.join(GCS_DS_PATH, 'images')
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMG_SIZE = 528

train_df = pd.read_csv(os.path.join(PROJECT_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_DIR, 'test.csv'))
train_df.drop([379, 1173], inplace=True)


def pathStr(img_id):
  return os.path.join(IMG_DIR, '{}.jpg'.format(img_id))


train_paths = train_df.image_id.apply(pathStr)
test_paths = test_df.image_id.apply(pathStr)
train_labels = train_df.loc[:, 'healthy':].values

def read_img(filename, label=None, image_size=(IMG_SIZE, IMG_SIZE)):
  bits = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(bits, channels=3)
  image = tf.cast(image, tf.float32) / 255.0
  image = tf.image.resize(image, image_size)
  if label is None:
    return image
  else:
    return image, label# {'h_output': label[0], 'm_output': label[1], 'r_output': label[2], 's_output': label[3]}

def _read_image_preprocessing(filename, resolution=IMG_SIZE):
  bits = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(bits, channels=3).numpy()
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)
  filter_status = False
  base = [blur, gray]
  for canny_base in base:
    canny = cv2.Canny(canny_base, 100, 200)
    x, y, w, h = cv2.boundingRect(canny)
    if w >= resolution//2 and h >= resolution//2:
      filter_status = True
      break
  if not filter_status:
    pass
  else:
    image = image[y:y+h, x:x+w]
  image = cv2.resize(image, (resolution, resolution))
  image = image.astype(np.float32) / 255.0
  return image

def read_image_preprocessing(filename, label=None):
  image = tf.py_function(_read_image_preprocessing, [filename], tf.float32)
  if label is None:
    return image
  else:
    return image, label


def data_augment(image, label=None, seed=42):
  image = tf.image.random_flip_left_right(image, seed=seed)
  image = tf.image.random_flip_up_down(image, seed=seed)
  image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
  if label is None:
    return image
  else:
    return image, label

train_dataset = (
      tf.data.Dataset
          .from_tensor_slices((train_paths, train_labels))
          .map(read_img, num_parallel_calls=AUTO)
          .map(data_augment, num_parallel_calls=AUTO)
          .repeat()
          .shuffle(512)
          .batch(BATCH_SIZE)
          .prefetch(AUTO)
  )

test_dataset = (
      tf.data.Dataset
          .from_tensor_slices(test_paths)
          .map(read_img, num_parallel_calls=AUTO)
          .batch(BATCH_SIZE)
  )


def lr_scheduler_config(epoch_num):
    if epoch_num < 15:
        lr = 0.001 * (2 ** int((-epoch_num) / 5))
    else:
        lr = 0.001 * (2 ** int(-15 / 5)) * (3 ** int((-epoch_num + 15) / 3))
    lr = max(0.000001, lr)
    '''
    print('[Learning Rate]: {}'.format(lr))
    '''
    return lr


lr_scheduler_cosine = tf.keras.experimental.CosineDecayRestarts(0.001, 10, m_mul=1)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler_config)

learning_rate_history = []


class LearningRateTracker(tf.keras.callbacks.Callback):

    def __init__(self):
        super(LearningRateTracker, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        global learning_rate_history
        learning_rate_history.append(lr)
        print('[Learning Rate]: ', lr)

def create_model(model_name, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
  if model_name == 'efn_b4':
    model = efn.EfficientNetB4(weights=None, classes=4)
  elif model_name == 'efn_b4_p':
    model = tf.keras.models.Sequential()
    model.add(efn.EfficientNetB4(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'efn_b5_p':
    model = tf.keras.models.Sequential()
    model.add(efn.EfficientNetB5(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'efn_b6_p':
    model = tf.keras.models.Sequential()
    model.add(efn.EfficientNetB6(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'efn_b7_p':
    model = tf.keras.models.Sequential()
    model.add(efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'densenet121_p':
    model = tf.keras.models.Sequential()
    model.add(DenseNet121(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'densenet201_p':
    model = tf.keras.models.Sequential()
    model.add(DenseNet201(input_shape=input_shape, weights='imagenet', include_top=False))
  elif model_name == 'inceptionResV2_p':
    model = tf.keras.models.Sequential()
    model.add(InceptionResNetV2(input_shape=input_shape, weights='imagenet', include_top=False))
  if model_name.split('_')[-1] == 'p':
    model.add(GlobalAveragePooling2D())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))
  model.summary()
  return model

with strategy.scope():
    model = create_model('efn_b7_p')
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(label_smoothing=0), metrics=['categorical_accuracy'])

history = model.fit(
    train_dataset,
    epochs=40,
    steps_per_epoch=train_labels.shape[0]//BATCH_SIZE,
    callbacks=[lr_scheduler, LearningRateTracker()],
)

result = model.predict(test_dataset)
test_id = test_df.image_id.values
def submit(result):
  id_test = np.expand_dims(test_id, -1)
  df = pd.DataFrame(np.hstack((id_test, result)), columns=['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
  df.to_csv('submission.csv', index=False)

submit(result)