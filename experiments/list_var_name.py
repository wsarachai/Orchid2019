import re
import tensorflow as tf


def _sort(x, d):
    if re.match(r'MobilenetV2/Conv/', x) is not None:
        return 100
    if re.match(r'MobilenetV2/expanded_conv/', x) is not None:
        return 200
    if re.match(r'MobilenetV2/Conv_1/', x) is not None:
        return 20000
    if re.match(r'MobilenetV2/Logits/', x) is not None:
        return 30000
    key = re.search(r'[_-]([0-9]+)/', x)
    if key is not None:
        p = key.span()
        ret = (int(x[p[0]+1:p[1]-1])+1) * d
        ret = ret + _sort(x[p[1]:], d/1000)
    else:
        ret = 0
    return ret


def sort_fn(x):
    ret = _sort(x[0], 1000)
    return ret


def printv1():
    reader = tf.compat.v1.train.NewCheckpointReader('/Volumes/Data/tmp/orchids-models/mobilenet_v2_140_orchids52_0001/model.ckpt-10001')
    var_to_shape_map = reader.get_variable_to_shape_map()

    key_to_numpy = {}
    for key in sorted(var_to_shape_map.items(), key=sort_fn):
        key_to_numpy.update({key[0]: key[1]})

    keys = [
        'weights',
        'depthwise_weights',
        'BatchNorm/beta',
        'BatchNorm/gamma',
        'BatchNorm/moving_variance',
        'BatchNorm/moving_mean'
    ]
    expds = [
        'expand', 'depthwise', 'project'
    ]

    for k1 in keys:
        for _key in key_to_numpy:
            if _key.startswith('MobilenetV2/Conv/{}'.format(k1)):
                print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                break

    for i in range(0, 17):
        for sub in expds:
            for k2 in keys:
                if i == 0:
                    s_search = 'MobilenetV2/expanded_conv/{}/{}'.format(sub, k2)
                else:
                    s_search = 'MobilenetV2/expanded_conv_{}/{}/{}'.format(i, sub, k2)
                for _key in key_to_numpy:
                    if _key.startswith(s_search):
                        print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                        break

    for k1 in keys:
        for _key in key_to_numpy:
            if _key.startswith('MobilenetV2/Conv_1/{}'.format(k1)):
                print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                break

    for _key in key_to_numpy:
        print("{}: {}".format(_key, key_to_numpy[_key]))


def printv2():
    reader = tf.compat.v1.train.NewCheckpointReader(
        '/Volumes/Data/tmp/orchids-models/orchids2019/mobilenet_v2_140/pretrain1/ckpt-2')
    var_to_shape_map = reader.get_variable_to_shape_map()

    key_to_numpy = {}
    for key in sorted(var_to_shape_map.items()):
        key_to_numpy.update({key[0]: key[1]})
        #print("{}: {}".format(key[0], key[1]))

    keys = [
        'kernel',
        'depthwise_kernel',
        'beta',
        'gamma',
        'moving_variance',
        'moving_mean'
    ]

    for x in range(0, 1):
        for y in range(0, 104):
            for k in keys:
                s_search = 'model/layer_with_weights-{}/layer_with_weights-{}/{}'.format(x, y, k)
                for _key in key_to_numpy:
                    if _key.startswith(s_search):
                        print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                        break

    for k in ['bias', 'bias', 'kernel', 'kernel']:
        s_search = 'model/layer_with_weights-1/dense/{}'.format(k)
        for _key in key_to_numpy:
            if _key.startswith(s_search):
                print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                break

    for _key in key_to_numpy:
        print("{}: {}".format(_key, key_to_numpy[_key]))


if __name__ == '__main__':
    # tf.constant(reader.get_tensor(key[0]))
    printv2()
