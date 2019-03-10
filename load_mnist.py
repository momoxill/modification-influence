# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import gzip
from numpy import NaN
import numpy as np
from sklearn import preprocessing

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet
import pandas as pd
from scipy.sparse import coo_matrix, bmat,csc_matrix
from sklearn.model_selection import train_test_split


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 np array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D unit8 np array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 np array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D unit8 np array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


# def load_mnist(train_dir, validation_size=5000):
#
#   SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
#
#   TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
#   TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
#   TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
#   TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
#
#   local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
#                                    SOURCE_URL + TRAIN_IMAGES)
#   with open(local_file, 'rb') as f:
#     train_images = extract_images(f)
#
#   local_file = base.maybe_download(TRAIN_LABELS, train_dir,
#                                    SOURCE_URL + TRAIN_LABELS)
#   with open(local_file, 'rb') as f:
#     train_labels = extract_labels(f)
#
#   local_file = base.maybe_download(TEST_IMAGES, train_dir,
#                                    SOURCE_URL + TEST_IMAGES)
#   with open(local_file, 'rb') as f:
#     test_images = extract_images(f)
#
#   local_file = base.maybe_download(TEST_LABELS, train_dir,
#                                    SOURCE_URL + TEST_LABELS)
#   with open(local_file, 'rb') as f:
#     test_labels = extract_labels(f)
#
#   if not 0 <= validation_size <= len(train_images):
#     raise ValueError(
#         'Validation size should be between 0 and {}. Received: {}.'
#         .format(len(train_images), validation_size))
#
#   validation_images = train_images[:validation_size]
#   validation_labels = train_labels[:validation_size]
#   train_images = train_images[validation_size:]
#   train_labels = train_labels[validation_size:]
#
#   train_images = train_images.astype(np.float32) / 255
#   validation_images = validation_images.astype(np.float32) / 255
#   test_images = test_images.astype(np.float32) / 255
#
#   train = DataSet(train_images, train_labels)
#   validation = DataSet(validation_images, validation_labels)
#   test = DataSet(test_images, test_labels)
#
#   return base.Datasets(train=train, validation=validation, test=test)


def load_mnist(local_url, validation_size=1000):
    data = pd.read_csv(local_url, dtype='object')
    data = data.drop_duplicates()
    # time
    data['INST_DATE'] = pd.to_datetime(data['INST_DATE'].str.strip().str.split(' ').str[0], errors='ignore')
    data['DETECT_DATE'] = pd.to_datetime(data['DETECT_DATE'].str.strip().str.split(' ').str[0], errors='ignore')
    data['FAULT_DATE'] = pd.to_datetime(data['FAULT_DATE'].str.strip().str.split(' ').str[0], errors='ignore')
    data['save'] = data['INST_DATE'] - data['DETECT_DATE']
    data['save'] = [x.days / 30 if not pd.isnull(x) else np.nan for x in data['save']]
    data['work'] = data['FAULT_DATE'] - data['DETECT_DATE']
    data['work'] = [x.days / 30 if not pd.isnull(x) else np.nan for x in data['work']]
    data['FAULT_MONTH'] = [x.month for x in data['FAULT_DATE']]
    data['FAULT_MONTH'] = data['FAULT_MONTH'].values.astype('str')
    # min_max_scaler
    data = data[data['save'] >= 0]
    data = data[data['work'] >= 0]

    # 1-1
    data = data.drop_duplicates(subset=['ORG_NO', 'SPEC_CODE', 'COMM_MODE', 'MANUFACTURER', 'ARRIVE_BATCH_NO',
                                        'FAULT_MONTH', 'work', 'save'], keep=False)
    print('drop_duplicates-1-1 \n', len(data))

    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data['work'] = min_max_scaler.fit_transform(data['work'].reshape(-1, 1))
    data['save'] = min_max_scaler.fit_transform(data['save'].reshape(-1, 1))
    # data.drop(['work', 'save'],axis=1, inplace=True)
    # select 4 fault
    data['FAULT_TYPE_1'] = [x[0:2] for x in data['FAULT_TYPE'].values.astype('str')]
    data['FAULT_TYPE_3'] = [x[0:4] for x in data['FAULT_TYPE'].values.astype('str')]
    data = data[data['FAULT_TYPE_1'] == '04']
    data = data[(data['FAULT_TYPE_3'] != '0412') & (data['FAULT_TYPE'] != '04')]
    data = data[(data['FAULT_TYPE'] != '04')]
    # data = data[(data['FAULT_TYPE_3'] != '0404') | (data['FAULT_TYPE'] != '0409')]

    # select 10 sort
    data = data[data['SORT_CODE'] == '10']
    data.drop(['SORT_CODE'], axis=1, inplace=True)

    data.drop(
        ['EQUIP_ID', 'SYNC_ORG_NO', 'INST_DATE', 'DETECT_DATE', 'FAULT_DATE', 'ORG_NAME', 'FAULT_TYPE', 'FAULT_TYPE_1'],
              axis=1, inplace=True)
    data['COMM_MODE'] = data['COMM_MODE'].fillna('9999')
    # encode
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    data['ORG_NO'] = le.fit_transform(data['ORG_NO'])
    data['SPEC_CODE'] = le.fit_transform(data['SPEC_CODE'])
    data['COMM_MODE'] = le.fit_transform(data['COMM_MODE'])
    data['ARRIVE_BATCH_NO'] = le.fit_transform(data['ARRIVE_BATCH_NO'])
    data['MANUFACTURER'] = le.fit_transform(data['MANUFACTURER'])
    data['FAULT_MONTH'] = le.fit_transform(data['FAULT_MONTH'])
    data['FAULT_TYPE_3'] = le.fit_transform(data['FAULT_TYPE_3'])

    data_X = data.drop(['FAULT_TYPE_3'], axis=1)
    data_Y = data['FAULT_TYPE_3']
    # data_X = csc_matrix(data_X)
    train_images, test_images, train_labels, test_labels = train_test_split(data_X, data_Y, test_size=0.30, random_state=27)

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


def load_small_mnist(train_dir, validation_size=5000, random_seed=0):
    np.random.seed(random_seed)
    data_sets = load_mnist(train_dir, validation_size)

    train_images = data_sets.train.x
    train_labels = data_sets.train.labels
    perm = np.arange(len(train_labels))
    np.random.shuffle(perm)
    num_to_keep = int(len(train_labels) / 10)
    perm = perm[:num_to_keep]
    train_images = train_images[perm, :]
    train_labels = train_labels[perm]

    validation_images = data_sets.validation.x
    validation_labels = data_sets.validation.labels
    # perm = np.arange(len(validation_labels))
    # np.random.shuffle(perm)
    # num_to_keep = int(len(validation_labels) / 10)
    # perm = perm[:num_to_keep]
    # validation_images = validation_images[perm, :]
    # validation_labels = validation_labels[perm]

    test_images = data_sets.test.x
    test_labels = data_sets.test.labels
    # perm = np.arange(len(test_labels))
    # np.random.shuffle(perm)
    # num_to_keep = int(len(test_labels) / 10)
    # perm = perm[:num_to_keep]
    # test_images = test_images[perm, :]
    # test_labels = test_labels[perm]

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)
