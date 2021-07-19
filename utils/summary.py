from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random

from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary
from tensorboard.plugins.hparams import summary_v2


class KSummary(object):
    def __init__(self, summary_path=None, target=1, image_logging_every=10):
        self._writer = None
        self.keep_last_update = True
        self.image_logging_every = image_logging_every
        self.summary_path = summary_path

        self.epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._current_step = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.target = tf.Variable(target, trainable=False, dtype=tf.int64)
        self.log_image_at_step = tf.Variable(random.randint(0, self.target), trainable=False, dtype=tf.int64)

    def re_init(self, summary_path, target):
        self._writer = None
        self.target.assign(target)
        self.summary_path = summary_path

    def hparams_pb(self, hparams):
        with self.get_writer().as_default():
            summary_v2.hparams_pb(hparams)

    def hparams(self, hparams):
        with self.get_writer().as_default():
            summary_v2.hparams(hparams)

    def session_end_pb(self):
        with self.get_writer().as_default():
            pb = summary.session_end_pb(api_pb2.STATUS_SUCCESS)
            raw_pb = pb.SerializeToString()
            tf.compat.v2.summary.experimental.write_raw_pb(raw_pb, step=0)

    def trace_on(self, graph):
        with self.get_writer().as_default():
            tf.summary.trace_on(graph=graph)

    def graph(self, fn_name, graph):
        with self.get_writer().as_default():
            try:
                # tf.summary.graph support only tf 2.5
                tf.summary.graph(graph)
            except:
                tf.summary.trace_export(name=fn_name, step=0)

    def get_writer(self):
        if self._writer is None:
            self.summary_path = self.summary_path if self.summary_path is not None else "/tmp/summary"
            self._writer = tf.summary.create_file_writer(self.summary_path)
        return self._writer

    def image_update(self, name, unit, max_outputs=3):
        if self._current_step == self.log_image_at_step:
            with self.get_writer().as_default():
                tf.summary.image(name, unit, self.epoch, max_outputs=max_outputs)

    def histogram_update(self, name, unit):
        if self._current_step == self.target:
            with self.get_writer().as_default():
                tf.summary.histogram(name, unit, self.epoch)

    def scalar_update(self, name, unit, step):
        with self.get_writer().as_default():
            tf.summary.scalar(name, unit, step)

    def set_epoch(self, epoch):
        self.epoch.assign(epoch)

    def end_step(self):
        self._current_step.assign_add(1)

    def end_epoch(self):
        self._current_step.assign(1)
        sel = tf.random.uniform(shape=[], minval=0, maxval=self.target, dtype=tf.int64, seed=10)
        self.log_image_at_step.assign(sel)


k_summary = KSummary()
