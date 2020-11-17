from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import lib_utils
from data import data_utils, orchids52_dataset
from nets import utils

train_step = 1
model_name = utils.MOBILENET_V2_140_ORCHIDS52
workspace_path = os.environ['WORKSPACE'] if 'WORKSPACE' in os.environ else '/Volumes/Data/tmp'
data_path = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else '/Volumes/Data/_dataset/_orchids_dataset'
data_dir = os.path.join(data_path, 'orchids52_data')
checkpoint_path = os.path.join(workspace_path, 'orchids-models', 'orchids2019', model_name)
print(model_name)
print(workspace_path)
print(data_path)
print(data_dir)
print(checkpoint_path)

batch_size = 32
if train_step > 1:
    batch_size = batch_size // 4
print(batch_size)

load_dataset = data_utils.dataset_mapping[data_utils.ORCHIDS52_V1_TFRECORD]
create_model = utils.nets_mapping[utils.MOBILENET_V2_140_ORCHIDS52]

test_ds = load_dataset(split="test", batch_size=batch_size, root_path=data_dir)
print(test_ds.size)

learning_rate = 0.01
training_step = utils.TRAIN_TEMPLATE.format(step=train_step)
learning_rate = lib_utils.config_learning_rate(learning_rate=learning_rate,
                                               exp_decay=False,
                                               training_step=training_step)
optimizer = lib_utils.config_optimizer(learning_rate, training_step=training_step)
loss_fn = lib_utils.config_loss()
print(training_step)
print(learning_rate)

model = create_model(num_classes=orchids52_dataset.NUM_OF_CLASSES,
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
train_model.evaluate(datasets=test_ds)
