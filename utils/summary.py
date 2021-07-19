from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary
from tensorboard.plugins.hparams import summary_v2


class KSummary(object):
    def __init__(self, summary_path=None):
        self._writer = None
        self.keep_last_update = True
        self.summary_path = summary_path
        self._image_maps = {}
        self._histrogram_maps = {}
        self._scalar_maps = {}

    def re_init(self, summary_path):
        self._writer = None
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

    def get_unit_key(self, name, unit):
        if hasattr(unit, "name"):
            u_name = unit.name
            if ":" in unit.name:
                u_name = unit.name.split(":")[0]
            key = "{}/{}".format(name, u_name)
        else:
            key = "{}".format(name)
        return key

    def image_update(self, name, unit, step=0, max_outputs=3):
        if self.keep_last_update:
            self._image_maps.update(
                {self.get_unit_key(name, unit): {"name": name, "unit": unit, "max_outputs": max_outputs}}
            )
        else:
            with self.get_writer().as_default():
                tf.summary.image(name, unit, step)

    def histogram_update(self, name, unit, step=0):
        if self.keep_last_update:
            self._histrogram_maps.update({self.get_unit_key(name, unit): {"name": name, "unit": unit}})
        else:
            with self.get_writer().as_default():
                tf.summary.histogram(name, unit, step)

    def scalar_update(self, name, unit, step=0):
        if self.keep_last_update:
            self._scalar_maps.update({self.get_unit_key(name, unit): {"name": name, "unit": unit}})
        else:
            with self.get_writer().as_default():
                tf.summary.scalar(name, unit, step)

    def flush_all(self, step):
        with self.get_writer().as_default():
            for k, v in self._image_maps.items():
                tf.summary.image(k, v["unit"], step, max_outputs=v["max_outputs"])
            for k, v in self._histrogram_maps.items():
                tf.summary.histogram(k, v["unit"], step)
            for k, v in self._scalar_maps.items():
                tf.summary.scalar(k, v["unit"], step)


k_summary = KSummary()
