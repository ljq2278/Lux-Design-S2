import json
import numpy as np


def _convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return 'shape_'+'_'.join([str(x) for x in list(obj.shape)])
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(elem) for elem in obj]
    else:
        return obj

def saveToJsonExample(my_object, file_name):
    # Open a file for writing
    with open('/home/ljq/PycharmProjects/Lux-Design-S2/kits/rl/sb3/data_example/'+file_name, 'w') as f:
        # Serialize the object to JSON, converting NumPy arrays to lists
        json.dump(_convert_numpy(my_object), f, indent=4)
