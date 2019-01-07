import torch
import torch.utils.data as data  
from PIL import Image

import os 
import math 
import functools
import json
import copy
import random

'''
All the data paths have been previously created by 'make_kinetics_datafile.py'
Train data list : train_datalist.json
Val data list : val_datalist.json
Each file contains a list whose members includes four atributes : 'path', 'label', 'class_name', 'num_frames'

Dataset hierarchy:
root path(videos_dataset/Kinetics)->
    kinetics_400_jpgs->
        kinetics-400_train->(contains 400 directories)
            abseiling->
                -3B32lodo2M_000059_000069->
                    image_00001.jpg
                    image_00002.jpg
                        ...
                ...
            ...
            ...
            zumba->

        :
        :
        kinetics-400_val->(contains 400 directories for validation)
            :
            :
        
        kinetics-400_test->
            test->(contains test data, labels have not been published)
'''



class Kinetics(data.Dataset):
    def __init__(self, root='./videos_dataset/Kinetics/kinetics_400_jpgs', 
                mode='train', sample_frames=64, transform=None):
        '''
        mode = ['train', 'va', 'test']
        '''

        # read data list file
        datalist_path = os.path.join(root, 'train_datalist.json' if mode=='train' else 'val_datalist.json')
        with open(datalist_path, 'r') as f:
            self.data_list = json.load(f)

        self.mode = mode
        self.sample_frames = sample_frames
        self.transform = transform

    def __getitem__(self, index):
        data = self.data_list[index]

        path = data['path']
        label = data['label']
        num_frames = data['num_frames']
        class_name = data['class_name']

        if self.mode == 'test':
            frame_indices = self._get_test_indices(data)
        else:
            frame_indices = self._get_indices(data)

        print(len(frame_indices))
        video = self._frames_loader(path, frame_indices) 
        # T, C, H, W
        print(len(video))

        if self.transform is not None:
            video = self.transform(video)
        
        return video, label

    def __len__(self):
        return len(self.data_list)

    def _get_indices(self, data_info):
        num_frames = data_info['num_frames']
        if num_frames <= self.sample_frames:
            indices = list(range(1, num_frames+1, 1))
            while len(indices) < self.sample_frames:
                indices.extend(range(1, num_frames+1, 1))

        else:
            offset = random.choice(range(1, num_frames-self.sample_frames+1, 1))
            indices = list(range(offset, self.sample_frames+1, 1))

        return indices[:self.sample_frames]

    def _get_test_indices(self, data_info):
        num_frames = data_info['num_frames']
        if num_frames <= self.sample_frames:
            indices = list(range(1, num_frames+1, 1))
            while len(indices) < self.sample_frames:
                indices.extend(range(1, num_frames+1, 1))

            return indices[:self.sample_frames]

        else:
            indices = list(range(1, num_frames+1, 1))

            return indices


    def _frames_loader(self, video_dir_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    with Image.open(f) as img:
                        img = img.convert('RGB')
                        video.append(img)

            else:
                assert False, 'something in frames path'

        return video

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.abspath('.'))
    print(sys.path)

    import utils.transforms as ut_transform
    train_set = Kinetics(transform=ut_transform.GroupStackToTensor())
    print('length of train dataset is :', len(train_set))
    print('first data size is :', train_set[0][0].size())