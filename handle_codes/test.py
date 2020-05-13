import json
import numpy as np

"""
# test_path = "/root/hdd/yankun/with_jameel/data/test.json"
test_path = "/root/hdd/yankun/with_jameel/data/test1.json"
test_file = open(test_path, 'w')
temp = dict()
aa = [1, 2, 3, 4, 5]
aa = np.asarray(aa, dtype = float)
temp['afa'] = list(aa)
# temp['afa'] = aa
test_file.write(json.dumps(temp) + '\n')
test_file.write(json.dumps(temp) + '\n')
"""
import numpy as np
import json

# input
glove_vector_path = "/root/hdd/yankun/data/glove.840B.300d.txt"
label_to_name_path = "/root/hdd/yankun/data/label_to_name.txt"
# output
label_to_vector_path = "/root/hdd/yankun/with_jameel/data/label_to_vector_glove.json"

glove_vector = dict()
label_to_names = dict()

# Prof.Jmaeel is familiar with the format of GLoVe dataset
with open(glove_vector_path, 'r') as gv_file:
    print("glove_vector is handling...")
    for line in gv_file:
        line = line.rstrip("\n\r")
        # line_name, line_vector = line.split(",")
        line = line.split(' ')
        line_name = line[0]
        line_vector = line[1:]
        line_vector = np.asarray(line_vector, dtype=float)
        glove_vector[line_name] = line_vector

    # none_vector = np.ones(300, dtype=float)
    print("glove_vector is done")
for key, value in glove_vector.items():
    if 'trash' in key and 'can' in key:
        print(key)

import pdb; pdb.set_trace()
