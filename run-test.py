from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import lib_utils
from data import data_utils
from nets import utils

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

flags.DEFINE_string('dataset', data_utils.ORCHIDS52_V1_TFRECORD,
                    'Dataset')

flags.DEFINE_string('model', utils.MOBILENET_V2_140_ORCHIDS52,
                    'Model')


def main(unused_argv):
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

    learning_rate = 0.01
    training_step = utils.TRAIN_TEMPLATE.format(step=FLAGS.train_step)
    learning_rate = lib_utils.config_learning_rate(learning_rate=learning_rate,
                                                   exp_decay=False,
                                                   training_step=training_step)
    optimizer = lib_utils.config_optimizer(learning_rate, training_step=training_step)
    loss_fn = lib_utils.config_loss()
    print(training_step)
    print(learning_rate)

    model = create_model(num_classes=test_ds.num_of_classes,
                         optimizer=optimizer,
                         loss_fn=loss_fn,
                         batch_size=batch_size,
                         step=training_step)

    train_model = lib_utils.TrainClassifier(model=model,
                                            batch_size=batch_size)

    model.config_checkpoint(checkpoint_path)
    epoch = model.restore_model_variables()

    model.summary()

    print('Test accuracy: ')
    train_model.evaluate(datasets=test_ds, bash=FLAGS.bash,)


if __name__ == '__main__':
    lib_utils.start(main)
