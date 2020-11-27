from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import collections
import tensorflow as tf
import numpy as np

from data import data_utils
from nets import utils
from nets.mobilenet_v2 import default_image_size

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

FLAGS = flags.FLAGS


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


def image_preprocessing_fn(image, height, width,
                           central_fraction=0.875, scope=None):
    with tf.compat.v1.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.compat.v1.image.resize_bilinear(image, [height, width],
                                                       align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def main(_):
    workspace_path = os.environ['WORKSPACE'] if 'WORKSPACE' in os.environ else '/Volumes/Data/tmp'
    data_path = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else '/Volumes/Data/_dataset/_orchids_dataset'
    data_dir = os.path.join(data_path, 'orchids52_data')
    load_dataset = data_utils.dataset_mapping[FLAGS.dataset]
    create_model = utils.nets_mapping[FLAGS.model]
    checkpoint_path = os.path.join(workspace_path, 'orchids-models', 'orchids2019', FLAGS.model, 'pretrain')

    with tf.Graph().as_default():
        dataset_images = create_image_lists(image_dir=FLAGS.image_dir)
        test_ds = load_dataset(split="test", batch_size=batch_size, root_path=data_dir)

        jpeg_data = tf.compat.v1.placeholder(tf.string, name='DecodeJPGInput')
        decoded_image = tf.compat.v1.image.decode_jpeg(jpeg_data, channels=3))

        eval_image_size = default_image_size
        image = image_preprocessing_fn(decoded_image, eval_image_size, eval_image_size)

        image = tf.expand_dims(image, 0)
        logits, end_points = network_fn(image, **kwargs)

        variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        softmax = tf.nn.softmax(logits, axis=1)
        softmax = tf.squeeze(softmax)

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        print(">>> Total number fo parameters: {}".format(custom_utils.get_total_parameters()))

        #
        # Create key maps
        #
        # reader = tf.train.NewCheckpointReader(checkpoint_path)
        # var_to_shape_map = reader.get_variable_to_shape_map()
        # key_to_numpy = {}
        # for key in var_to_shape_map:
        #   key_to_numpy[key] = reader.get_tensor(key)
        # key_to_tensor = {}
        # for var in variables_to_restore:
        #   key = var.op.name
        #   key_to_tensor[key] = var

        # get variable
        # for _k, var in key_to_numpy.items():
        #   filename = _k.replace('/', '-')
        #   filename = '/tmp/chk-weight-2/{}.txt'.format(filename)
        #   print("Writing checipoint weight {}".format(filename))
        #   var = np.expand_dims(var, axis=0)
        #   write_weight(filename, var)

        _init_fn = nets_factory.get_init_fn(
            model_name=FLAGS.model_name,
            is_training=False)

        with tf.Session() as sess:
            if _init_fn:
                init_callback_fn = _init_fn(var_list=variables_to_restore,
                                            checkpoint_path=checkpoint_path)
                sess.run(tf.compat.v1.global_variables_initializer())
                if init_callback_fn:
                    init_callback_fn(sess)
            else:
                saver = tf.train.Saver(variables_to_restore)
                sess.run(tf.compat.v1.global_variables_initializer())
                saver.restore(sess, checkpoint_path)

            # get variable
            # for _k, var in key_to_tensor.items():
            #   filename = _k.replace('/', '-')
            #   filename = '/tmp/m2-weight/{}.txt'.format(filename)
            #   print ("Writing {}".format(filename))
            #   v = sess.run(var)
            #   write_weight(filename, v)

            # fl = open("/Users/sarachaii/Documents/confusion-label.txt", "w")
            # fp = open("/Users/sarachaii/Documents/confusion-prediction.txt", "w")

            count = 0
            corrected = 0
            for label, data in dataset_images.items():
                for file in data['testing']:
                    filename = os.path.join(FLAGS.image_dir, data['dir'], file)
                    image_data = tf.gfile.FastGFile(filename, 'rb').read()
                    results = sess.run([predictions, softmax], feed_dict={jpeg_data: image_data})

                    # for __k, layer in end_points.items():
                    #  filename = __k.replace('/', '-')
                    #  results = sess.run(layer, feed_dict={jpeg_data: image_data})
                    #  write_weight(filename='/tmp/m1_output/{}.txt'.format(filename), w=results)

                    count += 1
                    predict = results[0][0]
                    confi = results[1][predict]
                    spredict = "n{:04d}".format(predict)
                    if spredict == label:
                        corrected += 1
                    sys.stdout.write("\r>> {}/{}: Predict: {}, expected: {}, confident: {:.4f}, acc: {:.4f}".format(
                        count, total_images, spredict, label, confi, corrected / count))
                    sys.stdout.flush()
                    # fl.write("{},".format(label))
                    # fp.write("{},".format(spredict))

            sys.stdout.write('\n\nDone evaluation -- epoch limit reached')
            sys.stdout.write('Accuracy: {:.4f}'.format(corrected / total_images))
            sys.stdout.flush()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
