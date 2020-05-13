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
"""
the format of label_to_name.txt is as bellow :
    each line is like:
    "n02110063 malamute, malemute, Alaskan malamute"
"""
with open(label_to_name_path, 'r') as ltn_file:
    print("label_to_names is handling...")
    for line in ltn_file:
        line = line.rstrip("\n\r")
        line_temp = line.split(' ')
        line_number = line_temp[0]
        _, line_names = line.split(line_number)
        line_names = line_names.strip(' ')
        line_names = line_names.split(',')

        for i in range(len(line_names)):
            # line_names[i] = line_names[i].strip(' ').lower()
            line_names[i] = line_names[i].strip(' ')
        line_names = np.asarray(line_names, dtype=str)
        label_to_names[line_number] = line_names

    print("label_to_names is done")

print("now, find vector representations from GloVe for each label")
result_file = open(label_to_vector_path, 'w')
name_keys = glove_vector.keys()
for label, names in label_to_names.items():
    print("label:{}".format(label))
    temp_result = dict()
    num_name = len(names)
    name_to_vector = np.zeros(300, dtype = float)
    name_flag = False
    name_len = 0
    for i in range(num_name):
        temp_names = names[i].split(' ')
        temp_names_len = len(temp_names)
        temp_name_vector = np.zeros(300, dtype = float)
        flag_valid = True
        for sub_name in temp_names:
            if sub_name not in name_keys:
                flag_valid = False
                break
            else:
                temp_name_vector += glove_vector[sub_name]
        if flag_valid:
            temp_name_vector = temp_name_vector/temp_names_len
            name_to_vector += temp_name_vector
            name_flag = True
            name_len += 1
    # import pdb; pdb.set_trace()
    if name_flag:
        name_to_vector = name_to_vector/name_len
    else:
        name_to_vector = glove_vector['UNK']
        # for test
        # print("unknown")
        # name_to_vector = [1,2 ,3,4 ,5 , 1,]

    temp_result[label] = list(name_to_vector)

    result_file.write(json.dumps(temp_result) + '\n')
    # import pdb; pdb.set_trace()

print("done")
