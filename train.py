from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import lib_utils

from pickle import dump
from data import data_utils, orchids52_dataset
from data.data_utils import dataset_mapping
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

flags.DEFINE_string('total_epochs', '100,200,200,200',
                    'Total epochs')

flags.DEFINE_integer('start_state', 1,
                     'Start state')

flags.DEFINE_integer('end_state', 2,
                     'End state')

flags.DEFINE_float('learning_rate', 0.001,
                   'Learning Rate')

flags.DEFINE_string('aug_method', 'fast',
                    'Augmentation Method')

flags.DEFINE_string('dataset', data_utils.ORCHIDS52_V1_TFRECORD,
                    'Dataset')

flags.DEFINE_string('model', utils.MOBILENET_V2_140_ORCHIDS52,
                    'Model')


def main(unused_argv):
    logging.debug(unused_argv)
    workspace_path = os.environ['WORKSPACE'] if 'WORKSPACE' in os.environ else '/Volumes/Data/tmp'
    data_path = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else '/Volumes/Data/_dataset/_orchids_dataset'
    data_dir = os.path.join(data_path, 'orchids52_data')
    load_dataset = dataset_mapping[FLAGS.dataset]
    create_model = utils.nets_mapping[FLAGS.model]
    checkpoint_path = os.path.join(workspace_path, 'orchids-models', 'orchids2019', FLAGS.model)

    if not tf.io.gfile.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    total_epochs = [int(e) for e in FLAGS.total_epochs.split(',')]
    num_state = FLAGS.end_state - FLAGS.start_state
    assert(num_state == len(total_epochs))
    for idx, train_step in enumerate(range(FLAGS.start_state, FLAGS.end_state)):
        if train_step == 1:
            batch_size = FLAGS.batch_size
        else:
            batch_size = FLAGS.batch_size // 4

        train_ds = load_dataset(split="train",
                                batch_size=batch_size,
                                root_path=data_dir,
                                aug_method=FLAGS.aug_method)
        validate_ds = load_dataset(split="validate", batch_size=batch_size, root_path=data_dir)
        test_ds = load_dataset(split="test", batch_size=batch_size, root_path=data_dir)

        training_step = utils.TRAIN_TEMPLATE.format(step=train_step)

        learning_rate = lib_utils.config_learning_rate(FLAGS.learning_rate,
                                                       FLAGS.exp_decay,
                                                       training_step=training_step)
        optimizer = lib_utils.config_optimizer(learning_rate, training_step=training_step)
        loss_fn = lib_utils.config_loss()

        model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             training=True,
                             batch_size=batch_size,
                             step=training_step)

        model.config_checkpoint(checkpoint_path)
        train_model = lib_utils.TrainClassifier(model=model,
                                                batch_size=batch_size)

        epoch = model.restore_model_variables()
        model.summary()

        history_fine = train_model.fit(initial_epoch=epoch,
                                       epoches=total_epochs[idx],
                                       train_ds=train_ds,
                                       validate_ds=validate_ds,
                                       checkpoint_path=checkpoint_path,
                                       bash=FLAGS.bash)

        with open('{}-history.pack'.format(training_step), 'wb') as handle:  # saving the history of the model
            dump(history_fine['history'], handle)

        print('Test accuracy: ')
        train_model.evaluate(datasets=test_ds)

        model_path = os.path.join(checkpoint_path, str(train_step))
        if not tf.io.gfile.exists(model_path):
            tf.io.gfile.mkdir(model_path)
        model.save(model_path)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)
    lib_utils.start(main)
