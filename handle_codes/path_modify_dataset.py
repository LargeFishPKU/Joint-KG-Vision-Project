import json

# source_file_path = "/root/hdd/yankun/data/miniImagenet/base_old.json"
# result_file_path = "/root/hdd/yankun/data/miniImagenet/base.json"
# source_file_path = "/root/hdd/yankun/data/miniImagenet/val_old.json"
# result_file_path = "/root/hdd/yankun/data/miniImagenet/val.json"
source_file_path = "/root/hdd/yankun/data/miniImagenet/novel_old.json"
result_file_path = "/root/hdd/yankun/data/miniImagenet/novel.json"

source_file = open(source_file_path, 'r')
source_data = json.load(source_file)

result_dict = dict()
result_dict['label_names'] = source_data['label_names']
result_dict['image_labels'] = source_data['image_labels']
image_names_list = source_data['image_names']

for i in range(len(image_names_list)):
    image_names_list[i] = image_names_list[i].replace("/mnt/lustre/caixiaocong/CKD/fewshot/datasets",
                                "/root/hdd/yankun/data")
result_dict['image_names'] = image_names_list

with open(result_file_path, 'w') as file:
    json.dump(result_dict, file)

# import pdb; pdb.set_trace()
