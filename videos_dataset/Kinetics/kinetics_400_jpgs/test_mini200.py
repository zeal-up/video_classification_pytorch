import json
import numpy as np 
import os, sys


###############################################################################
# create class_to_idx file
###############################################################################

f = open('./videos_dataset/Kinetics/kinetics_400_jpgs/val_datalist.json', 'r')
data_400 = json.load(f)
f.close()


count = 1
data_200 = []
f = open('./videos_dataset/Kinetics/kinetics_400_jpgs/val_ytid_list.txt', 'r')
for line in f:
    data_200.append(line.strip())
f.close()

newdata_200 = []
class_to_idx = {}
idx = 0
for info in data_400:
    id_400 = info['path'].split('/')[-1][:11]
    if id_400 in data_200:
        count+= 1
        newdata_200.append(info)
        if info['class_name'] not in class_to_idx.keys():
                class_to_idx[info['class_name']] = idx 
                idx += 1


print('{} samples in the kinetcis_200'.format(len(data_200)))
print('{} samples is avaliable currently'.format(count))
print('{} class is exist in the created file'.format(len(class_to_idx)))


f = open('./videos_dataset/Kinetics/kinetics_200_jpgs/class_to_idx.json', 'w')
json.dump(class_to_idx, f)
f.close()

# test class_to_idx file
f = open('./videos_dataset/Kinetics/kinetics_200_jpgs/class_to_idx.json', 'r')
class_to_idx = json.load(f)
f.close()
print('number of new class is:', len(class_to_idx))

###################################################################################



for file_name in ['val', 'train']:
    f = open('./videos_dataset/Kinetics/kinetics_400_jpgs/'+file_name+'_datalist.json', 'r')
    data_400 = json.load(f)
    f.close()

    data_200 = []
    f = open('./videos_dataset/Kinetics/kinetics_400_jpgs/'+file_name+'_ytid_list.txt', 'r')
    for line in f:
        data_200.append(line.strip())
    f.close()

    count = 0
    newdata_200 = []
    class_name = []
    for info in data_400:
        id_400 = info['path'].split('/')[-1][:11]
        if id_400 in data_200:
            count+= 1
            info['label'] = class_to_idx[info['class_name']]
            newdata_200.append(info)
            if info['class_name'] not in class_name:
                    class_name.append(info['class_name'])

    print('{} samples in the kinetcis_200'.format(len(data_200)))
    print('{} samples is avaliable currently'.format(count))
    print('{} class is exist in the created file'.format(len(class_name)))

    # f = open('./videos_dataset/Kinetics/kinetics_200_jpgs/'+file_name+'_datalist_200.json', 'w')
    # json.dump(newdata_200, f)
    # f.close()