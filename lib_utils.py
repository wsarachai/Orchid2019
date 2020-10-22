from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pathlib


def get_step_number(checkpoint_dir):
    idx = checkpoint_dir.index('.')
    step = int(checkpoint_dir[:idx][-4:])
    return step


def latest_checkpoint(checkpoint_path):
    file_path = pathlib.Path(checkpoint_path)
    file_list = list(file_path.glob('*.h5'))

    if len(file_list) > 0:
        max_step = 0
        for file in file_list:
            step = get_step_number(file.name)
            if max_step < step:
                max_step = step
                file_path = file
        return file_path, max_step
    return None, 0
