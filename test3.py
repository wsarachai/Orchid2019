from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import collections
import tensorflow as tf
from tensorflow import keras

from experiments import list_var_name
from nets.mobilenet_v2 import default_image_size, create_mobilenet_v1, create_mobilenet_v2, IMG_SHAPE_224

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


def main1(_, nets=None):
    workspace_path = os.environ['WORKSPACE'] if 'WORKSPACE' in os.environ else '/Volumes/Data/tmp'
    checkpoint_path = os.path.join(workspace_path, 'orchids-models', 'orchids2019', FLAGS.model, 'pretrain', 'chk')

    with tf.compat.v1.Graph().as_default():
        with tf.compat.v1.variable_scope('MobilenetV2'):
            dataset_images = create_image_lists(image_dir=FLAGS.image_dir)
            jpeg_data = tf.compat.v1.placeholder(tf.string, name='DecodeJPGInput')
            decoded_image = tf.compat.v1.image.decode_jpeg(jpeg_data, channels=3)
            eval_image_size = default_image_size
            image = image_preprocessing_fn(decoded_image, eval_image_size, eval_image_size)

            inputs = tf.expand_dims(image, 0)
            logits = create_mobilenet_v1(inputs, alpha=1.4, classes=52)

            predictions = tf.argmax(logits, 1)
            softmax = tf.nn.softmax(logits, axis=1)
            softmax = tf.squeeze(softmax)

            var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.VARIABLES)
            name_list_v1 = list_var_name.load_v1()
            _mapped = zip(name_list_v1, var_list)
            variables_to_restore = {}
            for k, v in _mapped:
                src_name = k[0]
                variables_to_restore.update({src_name: v})
            checkpoint_path = '/Volumes/Data/tmp/orchids-models/mobilenet_v2_140_orchids52_0001/pretrain2/model.ckpt-12000'
            reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
            var_to_shape_map = reader.get_variable_to_shape_map()

            key_to_numpy = {}
            for key in var_to_shape_map:
                key_to_numpy[key] = tf.constant(reader.get_tensor(key))

            assign_op = []
            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),
                                             reuse=tf.compat.v1.AUTO_REUSE):
                for _src_name, dst_var in variables_to_restore.items():
                    src_name = _src_name
                    if src_name in key_to_numpy:
                        print('Restore variable: {}'.format(dst_var.op.name))
                        # copy_var = tf.compat.v1.get_variable(dst_name)
                        init = key_to_numpy[src_name]
                        assign0 = tf.compat.v1.assign(dst_var, init)
                        with tf.control_dependencies(assign_op):
                            assign_op.append(assign0)
                    else:
                        raise ValueError("Can't find {}".format(src_name))

            def callback(sess):
                sess.run(assign_op)

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                if callback:
                    callback(sess)

                count = 0
                corrected = 0
                for label, data in dataset_images.items():
                    for file in data['testing']:
                        filename = os.path.join(FLAGS.image_dir, data['dir'], file)
                        image_data = tf.io.gfile.GFile(filename, 'rb').read()
                        results = sess.run([predictions, softmax], feed_dict={jpeg_data: image_data})

                        total_images = 739
                        count += 1
                        predict = results[0][0]
                        confi = results[1][predict]
                        spredict = "n{:04d}".format(predict)
                        if spredict == label:
                            corrected += 1
                        sys.stdout.write("\r>> {}/{}: Predict: {}, expected: {}, confident: {:.4f}, acc: {:.4f}".format(
                            count, total_images, spredict, label, confi, corrected / count))
                        sys.stdout.flush()

                sys.stdout.write('\n\nDone evaluation -- epoch limit reached')
                sys.stdout.write('Accuracy: {:.4f}'.format(corrected / total_images))
                sys.stdout.flush()


def main(_, nets=None):
    workspace_path = os.environ['WORKSPACE'] if 'WORKSPACE' in os.environ else '/Volumes/Data/tmp'
    save_path = os.path.join(workspace_path, 'orchids-models', 'orchids2019', 'mobilenet_v2_140', 'pretrain', 'chk')

    model = create_mobilenet_v1(alpha=1.4, classes=52)

    var_list = model.weights
    name_list_v1 = list_var_name.load_v1()
    _mapped = zip(name_list_v1, var_list)
    variables_to_restore = {}
    for k, v in _mapped:
        src_name = k[0]
        variables_to_restore.update({src_name: v})
    checkpoint_path = '/Volumes/Data/tmp/orchids-models/mobilenet_v2_140_orchids52_0001/pretrain2/model.ckpt-12000'
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    key_to_numpy = {}
    for key in var_to_shape_map:
        key_to_numpy[key] = tf.constant(reader.get_tensor(key))

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),
                                     reuse=tf.compat.v1.AUTO_REUSE):
        for _src_name, dst_var in variables_to_restore.items():
            src_name = _src_name
            if src_name in key_to_numpy:
                print('Restore variable: {}'.format(dst_var.name))
                init = key_to_numpy[src_name]
                dst_var.assign(init)
            else:
                raise ValueError("Can't find {}".format(src_name))

    #model.save_weights(save_path)

    dataset_images = create_image_lists(image_dir=FLAGS.image_dir)

    count = 0
    corrected = 0
    central_fraction = 0.875
    total_images = 739
    for label, data in dataset_images.items():
        for file in data['testing']:
            filename = os.path.join(FLAGS.image_dir, data['dir'], file)
            image_data = tf.io.gfile.GFile(filename, 'rb').read()
            image = tf.image.decode_jpeg(image_data, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.central_crop(image, central_fraction=central_fraction)
            image = tf.expand_dims(image, 0)
            image = tf.compat.v1.image.resize_bilinear(
                images=image,
                size=(224, 224),
                align_corners=False)
            image = tf.subtract(image, 0.5)
            image_input = tf.multiply(image, 2.0)

            results = model(image_input)

            predictions = tf.argmax(results, axis=1)
            softmax = tf.nn.softmax(results, axis=1)
            softmax = tf.squeeze(softmax)

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
