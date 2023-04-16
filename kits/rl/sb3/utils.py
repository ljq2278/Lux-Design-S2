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

# there is two role "Ann" and "John", they paly game with rule express with codes below:
# ```
# Ann give a number named Num_A
# while Num_A!=0:
#     John ask Ann to give another number to Num_A
#     Ann give a number to Num_A
# John say: you get the num!
# ```
# you konw the game rule?
#
#
#
# ok,lets level up the game. there is two role "Ann" and "John", they paly game with rule express with codes below:
# ```
# Ann give a sentence named Sent_A
# John comprehension the meaning of Sent_A, and get the intense named Inte_A
# while Inte_A is not about disease cure:
#     John say "I am a doctor and please ask the right question"
#     Ann give a new sentence to Sent_A
#     John comprehension the meaning of Sent_A, and get the intense named Inte_A
# John say: you get my point!
# ```
# you konw the game rule?