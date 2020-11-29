from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import collections
import tensorflow as tf
import numpy as np

import lib_utils
from data import data_utils
from nets import utils, constants
from nets.mobilenet_v2 import PreprocessLayer

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging
FLAGS = flags.FLAGS

flags.DEFINE_boolean('bash', False,
                     'Execute from bash')

flags.DEFINE_boolean('exp_decay', False,
                     'Exponential decay learning rate')

flags.DEFINE_integer('batch_size', 32,
                     'Batch size')

flags.DEFINE_integer('train_step', 1,
                     'Training step')

flags.DEFINE_float('learning_rate', 0.001,
                   'Learning Rate')

flags.DEFINE_string('dataset', data_utils.ORCHIDS52_V2_TFRECORD,
                    'Dataset')

flags.DEFINE_string('model', utils.MOBILENET_V2_140_ORCHIDS52,
                    'Model')

flags.DEFINE_string('checkpoint_path', None,
                    'Checkpoint path')

flags.DEFINE_string('image_dir',
                    '/Volumes/Data/tmp/orchids52_data/test/',
                    'The directory where the dataset images are locate')

flags.DEFINE_string('optimizer', 'rmsprop',
                    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
                    '"ftrl", "momentum", "sgd" or "rmsprop".')

flags.DEFINE_string('trained_path',
                    '/Volumes/Data/tmp/orchids-models/mobilenet_v2_140_orchids52_0001/pretrain2/model.ckpt-12000',
                    'Checkpoint Path')


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


def main1(unused_argv):
    logging.debug(unused_argv)
    workspace_path = os.environ['WORKSPACE'] if 'WORKSPACE' in os.environ else '/Volumes/Data/tmp'
    data_path = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else '/Volumes/Data/_dataset/_orchids_dataset'
    data_dir = os.path.join(data_path, 'orchids52_data')
    checkpoint_path = os.path.join(workspace_path, 'orchids-models', 'orchids2019', FLAGS.model)
    print('Model: {}'.format(FLAGS.model))
    print('Workspace: {}'.format(workspace_path))
    print('Data dir: {}'.format(data_dir))
    print('Checkpoint path: {}'.format(checkpoint_path))

    batch_size = 32
    if FLAGS.train_step > 1:
        batch_size = FLAGS.batch_size // 4
    print(batch_size)

    load_dataset = data_utils.dataset_mapping[FLAGS.dataset]
    create_model = utils.nets_mapping[FLAGS.model]

    test_ds = load_dataset(split="test", batch_size=batch_size, root_path=data_dir)
    print(test_ds.size)
    print(test_ds.num_of_classes)

    training_step = constants.TRAIN_TEMPLATE.format(step=FLAGS.train_step)
    print(training_step)

    model = create_model(num_classes=test_ds.num_of_classes,
                         optimizer=None,
                         loss_fn=None,
                         batch_size=batch_size,
                         step=training_step)

    train_model = lib_utils.TrainClassifier(model=model,
                                            batch_size=batch_size)

    model.restore_model_variables(
        checkpoint_path=FLAGS.checkpoint_path)
    model.summary()

    print('Test accuracy: ')
    train_model.evaluate(datasets=test_ds, bash=FLAGS.bash, )


def main(unused_argv):
    logging.debug(unused_argv)
    load_dataset = data_utils.dataset_mapping[FLAGS.dataset]
    create_model = utils.nets_mapping[FLAGS.model]

    training_step = constants.TRAIN_TEMPLATE.format(step=FLAGS.train_step)
    model = create_model(num_classes=load_dataset.num_of_classes,
                         optimizer=None,
                         loss_fn=None,
                         batch_size=1,
                         alpha=1.4,
                         include_top=True,
                         step=training_step)

    model.summary()

    count = 0
    corrected = 0
    total_images = load_dataset.test_size
    dataset_images = create_image_lists(image_dir=FLAGS.image_dir)
    preprocess_layer = PreprocessLayer(width=create_model.width,
                                       height=create_model.height)

    for label, data in dataset_images.items():
        for file in data['testing']:
            filename = os.path.join(FLAGS.image_dir, data['dir'], file)
            image_data = tf.io.gfile.GFile(filename, 'rb').read()
            image = tf.image.decode_jpeg(image_data, channels=3)
            image = tf.expand_dims(image, 0)
            image = preprocess_layer(image)
            result = model.process_step(image)

            count += 1
            predict = np.argmax(result, axis=1)[0]
            confident = result[0][predict]
            predict_str = "n{:04d}".format(predict)
            if predict_str == label:
                corrected += 1
            sys.stdout.write("\r>> {}/{}: Predict: {}, expected: {}, confident: {:.4f}, acc: {:.4f}".format(
                count, total_images, predict_str, label, confident, corrected / count))
            sys.stdout.flush()

        sys.stdout.write('\n\nDone evaluation -- epoch limit reached')
        sys.stdout.write('Accuracy: {:.4f}'.format(corrected / total_images))
        sys.stdout.flush()


if __name__ == '__main__':
    lib_utils.start(main)
