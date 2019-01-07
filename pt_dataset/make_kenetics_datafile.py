import os
import sys
import json


def make_classToindex(root_path):
    label = 0
    class_to_index = {}
    train_path = os.path.join(root_path, 'kinetics-400_train')
    for class_name in os.listdir(train_path):
        class_path = os.path.join(train_path, class_name)
        if not os.path.isdir(class_path):
            continue
        class_to_index[class_name] = label
        label += 1

    if len(class_to_index) != 400:
        assert False, "something error"
    
    # write to file
    file_name = os.path.join(root_path, 'class_index.json')
    with open(file_name, 'w') as f:
        json.dump(class_to_index, f)
    
    print('make class to index file successed')

def make_data_file(root_path, split):
    '''
    split = ['train', 'val']
    '''
    
    if not os.path.exists(os.path.join(root_path, 'class_index.json')):
        assert False, "must first make the class to index file"
    with open(os.path.join(root_path, 'class_index.json'), 'r') as f:
        class_to_index = json.load(f)

    data = []

    split_path = os.path.join(root_path, 'kinetics-400_'+split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        label = class_to_index[class_name]

        for video_id in os.listdir(class_path):
            video_path = os.path.join(class_path, video_id)
            if not os.path.isdir(video_path):
                continue
            
            num_frames = 0
            for image_file_name in os.listdir(video_path):
                
                if 'image' not in image_file_name:
                    continue
                num_frames += 1
        
            log = {}
            log['path'] = video_path
            log['label'] = label
            log['class_name'] = class_name
            log['num_frames'] = num_frames

            if num_frames != 0:
                data.append(log)

    with open(os.path.join(root_path, split+'_datalist.json'), 'w') as f:
        json.dump(data, f)

    data_list_length = len(data)
    print(split, '_datalist.json file create successfully, total files number:', data_list_length)

if __name__ == '__main__':
    # make_classToindex('./Kinetics/kinetics_400_jpgs')
    make_data_file('./videos_dataset/Kinetics/kinetics_400_jpgs', 'train')
    make_data_file('./videos_dataset/Kinetics/kinetics_400_jpgs', 'val')
            


            
