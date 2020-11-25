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

    keys = ['beta', 'gamma', 'variance', 'mean']
    while True:
        found = False
        for _key in key_to_numpy:
            if 'Conv/weights' in _key:
                #print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                print("{}:{}".format("weights", key_to_numpy.pop(_key)))
                found = True
                break
        if found:
            for k in keys:
                for _key in key_to_numpy:
                    if k in _key:
                        #print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                        print("{}:{}".format(k, key_to_numpy.pop(_key)))
                        found = True
                        break
        else:
            for _key in key_to_numpy:
                if 'depthwise/depthwise_weights' in _key:
                    #print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                    print("{}:{}".format("depthwise", key_to_numpy.pop(_key)))
                    found = True
                    break
            if found:
                for k in keys:
                    for _key in key_to_numpy:
                        if 'depthwise' in _key and k in _key:
                            #print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                            print("{}:{}".format(k, key_to_numpy.pop(_key)))
                            found = True
                            break
            found_prj = False
            for _key in key_to_numpy:
                if 'project/weights' in _key:
                    #print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                    print("{}:{}".format("project", key_to_numpy.pop(_key)))
                    found_prj = True
                    break
            if found_prj:
                for k in keys:
                    for _key in key_to_numpy:
                        if 'project' in _key and k in _key:
                            #print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                            print("{}:{}".format(k, key_to_numpy.pop(_key)))
                            found = True
                            break
            found = found or found_prj
        if not found:
            break

    for _key in key_to_numpy:
        print("{}: {}".format(_key, key_to_numpy[_key]))


def printv2():
    reader = tf.compat.v1.train.NewCheckpointReader(
        '/Volumes/Data/tmp/orchids-models/orchids2019/mobilenet_v2_140_orchids52/pretrain1/ckpt-100')
    var_to_shape_map = reader.get_variable_to_shape_map()

    key_to_numpy = {}
    for key in sorted(var_to_shape_map.items(), key=sort_fn):
        key_to_numpy.update({key[0]: key[1]})

    keys = ['beta', 'gamma', 'variance', 'mean']
    while True:
        found = False
        for _key in key_to_numpy:
            if 'kernel' in _key:
                #print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                print("{}:{}".format("weight", key_to_numpy.pop(_key)))
                found = True
                break
        if found:
            for k in keys:
                for _key in key_to_numpy:
                    if k in _key:
                        # print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                        print("{}:{}".format(k, key_to_numpy.pop(_key)))
                        found = True
                        break
        if not found:
            for _key in key_to_numpy:
                if 'depthwise_kernel' in _key:
                    #print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                    print("{}:{}".format("depthwise", key_to_numpy.pop(_key)))
                    found = True
                    break
            if found:
                for k in keys:
                    for _key in key_to_numpy:
                        if k in _key:
                            #print("{}:{}".format(_key, key_to_numpy.pop(_key)))
                            print("{}:{}".format(k, key_to_numpy.pop(_key)))
                            found = True
                            break
        if not found:
            break

    for _key in key_to_numpy:
        print("{}: {}".format(_key, key_to_numpy[_key]))


if __name__ == '__main__':
    # tf.constant(reader.get_tensor(key[0]))
    printv1()
