from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import collections
import tensorflow as tf
from tensorflow import keras

from nets.mobilenet_v2 import PredictionLayer, create_mobilenet_v2_custom, PreprocessLayer
from nets.mobilenet_v2_140 import load_trained_weights
from preprocesing.inception_preprocessing import preprocess_image

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset_name',
    'orchids52',
    'The name of the dataset to load.')

flags.DEFINE_string(
    'dataset_split_name',
    'test',
    'The name of the train/test split.')

flags.DEFINE_string(
    'dataset_dir',
    '/Volumes/Data/tmp/orchids52_data',
    'The directory where the dataset files are stored.')

flags.DEFINE_string(
    'model_name',
    'mobilenet_v2_140_stn_v12',
    'The name of the architecture to evaluate.')

flags.DEFINE_string(
    'step',
    'pretrain2',
    'The name of training steps.')

flags.DEFINE_string(
    'checkpoint_path',
    '/Volumes/Data/tmp/orchids-models/mobilenet_v2_140_stn_v12_orchids52_0001/pretrain2/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

flags.DEFINE_string(
    'image_dir',
    '/Volumes/Data/_dataset/_orchids_dataset/_images/orchids52/bak-01/test/',
    'The directory where the dataset images are locate')

flags.DEFINE_integer('total_test_images', 739, 'The total number of test images.')


def create_image_lists(image_dir):
    if not tf.io.gfile.exists(image_dir):
        logging.error("Image directory '" + image_dir + "' not found.")
        return None

    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.io.gfile.walk(image_dir))

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
                                for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'png']))

        file_list = []
        dir_name = os.path.basename(sub_dir[:-1] if sub_dir.endswith('/') else sub_dir)

        if dir_name == image_dir:
            continue
        logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.io.gfile.glob(file_glob))
        if not file_list:
            logging.warning('No files found')
            continue

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        testing_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            testing_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'testing': testing_images
        }

    return result


def get_data_files(data_sources):
    if isinstance(data_sources, (list, tuple)):
        data_files = []
        for source in data_sources:
            data_files += get_data_files(source)
    else:
        if '*' in data_sources or '?' in data_sources or '[' in data_sources:
            data_files = tf.io.gfile.glob(data_sources)
        else:
            data_files = [data_sources]
    if not data_files:
        raise ValueError('No data files found in %s' % (data_sources,))
    return data_files


def image_preprocessing_fn(image,
                           height, width,
                           central_fraction=0.875, scope=None):
    with tf.compat.v1.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)
        if height and width:
            image = tf.expand_dims(image, 0)
            image = tf.compat.v1.image.resize_bilinear(
                image,
                [height, width],
                align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def main(_):
    num_classes = 52
    inputs = keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32)
    model = create_mobilenet_v2_custom(input_shape=(224, 224, 3), alpha=1.4, classes=num_classes)
    prediction_layer = PredictionLayer(num_classes=num_classes)

    is_training = False
    x = model(inputs, training=is_training)
    output = prediction_layer(x, training=is_training)

    model = keras.Model(inputs, output, trainable=is_training)
    preprocess_layer = PreprocessLayer(width=224, height=224)
    dataset_images = create_image_lists(image_dir=FLAGS.image_dir)

    load_trained_weights(model)

    count = 0
    corrected = 0
    total_images = 739
    for label, data in dataset_images.items():
        for file in data['testing']:
            filename = os.path.join(FLAGS.image_dir, data['dir'], file)
            image_data = tf.io.gfile.GFile(filename, 'rb').read()
            image = tf.image.decode_jpeg(image_data, channels=3)
            image = tf.expand_dims(image, 0)
            image = preprocess_layer(image, training=False)
            results = model(image)

            predictions = tf.argmax(results, axis=1)
            softmax = tf.squeeze(results)

            count += 1
            predict = predictions[0]
            confident = softmax[predict]
            spredict = "n{:04d}".format(predict)
            if spredict == label:
                corrected += 1
            sys.stdout.write("\r>> {}/{}: Predict: {}, expected: {}, confident: {:.4f}, acc: {:.4f}".format(
                count, total_images, spredict, label, confident, corrected / count))
            sys.stdout.flush()

    sys.stdout.write('\n\nDone evaluation -- epoch limit reached')
    sys.stdout.write('Accuracy: {:.4f}'.format(corrected / total_images))
    sys.stdout.flush()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.app.run()
