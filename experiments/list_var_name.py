import re
import os
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


def load_v1():
    reader = tf.compat.v1.train.NewCheckpointReader('/Volumes/Data/tmp/orchids-models/mobilenet_v2_140_orchids52_0001/pretrain2/model.ckpt-12000')
    var_to_shape_map = reader.get_variable_to_shape_map()

    name_list = []
    key_to_numpy = {}
    for key in sorted(var_to_shape_map.items(), key=sort_fn):
        if 'RMSProp' not in key[0] and 'global_step' not in key[0]:
            key_to_numpy.update({key[0]: key[1]})

    keys = [
        'weights',
        'depthwise_weights',
        'BatchNorm/gamma',
        'BatchNorm/beta',
        'BatchNorm/moving_mean',
        'BatchNorm/moving_variance'
    ]
    expds = [
        'expand', 'depthwise', 'project'
    ]

    for k1 in keys:
        for _key in key_to_numpy:
            if _key.startswith('MobilenetV2/Conv/{}'.format(k1)):
                # print("{}:{}".format(_key, key_to_numpy[_key]))
                #name_list.append((_key, key_to_numpy.pop(_key)))
                name_list.append((_key, reader.get_tensor(_key)))
                key_to_numpy.pop(_key)
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
                        # print("{}:{}".format(_key, key_to_numpy[_key]))
                        name_list.append((_key, reader.get_tensor(_key)))
                        key_to_numpy.pop(_key)
                        break

    for k1 in keys:
        for _key in key_to_numpy:
            if _key.startswith('MobilenetV2/Conv_1/{}'.format(k1)):
                # print("{}:{}".format(_key, key_to_numpy[_key]))
                name_list.append((_key, reader.get_tensor(_key)))
                key_to_numpy.pop(_key)
                break

    for k1 in ['weight', 'bias']:
        for _key in key_to_numpy:
            if k1 in _key:
                # print("{}:{}".format(_key, key_to_numpy[_key]))
                name_list.append((_key, reader.get_tensor(_key)))
                key_to_numpy.pop(_key)
                break

    print(len(name_list))
    return name_list


def printv2():
    workspace_path = os.environ['WORKSPACE'] if 'WORKSPACE' in os.environ else '/Volumes/Data/tmp'
    checkpoint_path = os.path.join(workspace_path, 'orchids-models', 'orchids2019', 'mobilenet_v2_140', 'pretrain1')

    reader = tf.compat.v1.train.NewCheckpointReader(
        checkpoint_path + '/ckpt-100')
    var_to_shape_map = reader.get_variable_to_shape_map()

    name_list = []
    key_to_numpy = {}
    for key in sorted(var_to_shape_map.items()):
        if 'optimizer' not in key[0]:
            key_to_numpy.update({key[0]: key[1]})
            #print("{}: {}".format(key[0], key[1]))

    keys = [
        'kernel',
        'depthwise_kernel',
        'beta',
        'gamma',
        'moving_mean',
        'moving_variance'
    ]

    for x in range(0, 1):
        for y in range(0, 104):
            for k in keys:
                s_search = 'model/layer_with_weights-{}/layer_with_weights-{}/{}'.format(x, y, k)
                for _key in key_to_numpy:
                    if _key.startswith(s_search):
                        # print("{}:{}".format(_key, key_to_numpy[_key]))
                        name_list.append((_key, key_to_numpy.pop(_key)))
                        break

    for k in ['bias', 'bias', 'kernel', 'kernel']:
        s_search = 'model/layer_with_weights-1/dense/{}'.format(k)
        for _key in key_to_numpy:
            if _key.startswith(s_search):
                # print("{}:{}".format(_key, key_to_numpy[_key]))
                name_list.append((_key, key_to_numpy.pop(_key)))
                break

    # for _key in key_to_numpy:
    #     print("{}: {}".format(_key, key_to_numpy[_key]))

    print(len(name_list))
    return name_list


if __name__ == '__main__':
    load_v1()
    printv2()
