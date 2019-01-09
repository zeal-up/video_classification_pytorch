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

'''
这个文件用来载入连续的frames，可以是连续（I3D方法），然后再等间隔抽样（non-local方法）
所有的目录结构都已经提前写入了数据文件

ps：目前只支持kinetic 和 ucf101 的rgb数据载入
'''

class Consecutive(data.Dataset):
    def __init__(self, dataset='kinetics', split=1, train=True, 
                 sample_frames=64, interval=1, transform=None, test_mode='i3d'):
        '''
        Kinetics: root = './videos_dataset/Kinetics/kinetics_400_jpgs'
        Ucf101: root = './videos_dataset/UCF101/jpegs_256'
        split : only used in ucf101
        interval : sampling interval for consecutive frames.(the final sampled frames will be sample_frames/interval)
        test_mode : ['i3d', 'non_local', 'same']. For 'i3d', the whole video will be loaded. For 'non_local', it will evenly sample
                    10 clips for every videos. For 'same', the loading method will be the same as in training.
        '''

        # read data list file
        
        if dataset == 'kinetics':
            self.root = './videos_dataset/Kinetics/kinetics_400_jpgs' 
            datalist_file = 'train_datalist.json' if train else 'val_datalist.json'
        elif dataset == 'ucf101':
            self.root = './videos_dataset/UCF101/'
            datalist_file = 'trainlist{:02d}.json'.format(split) if train else 'testlist{:02d}.json'.format(split)
        else:
            assert False, 'only support kinetics and ucf101 right now'
        
        datalist_path = os.path.join(self.root, datalist_file)

        with open(datalist_path, 'r') as f:
            self.data_list = json.load(f)

        self.train = train
        self.dataset = dataset
        self.sample_frames = sample_frames
        self.interval = interval
        self.transform = transform
        self.test_mode = test_mode
        self.num_classes = {'kinetics':400, 'ucf101':101}[dataset]

    def __getitem__(self, index):
        data = self.data_list[index]

        path = data['path']
        label = data['label']
        num_frames = data['num_frames']
        class_name = data['class_name']

        if self.train:
            frame_indices = self._get_indices(data)
        elif self.test_mode == 'i3d':
            frame_indices = self._get_whole_indices(data)
        elif self.test_mode == 'non_local':
            clips_num = 10
            frame_indices = self._get_clips_indices(data, clips_num)
        else:
            frame_indices = self._get_indices(data)

        # print(len(frame_indices))
        # print(frame_indices)

        video = self._frames_loader(path, frame_indices) 
        # # T, C, H, W
        # # print(len(video))

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
                # print('looping video frames')

        else:
            offset = random.choice(range(1, num_frames-self.sample_frames+1, 1))
            indices = list(range(offset, offset+self.sample_frames, 1))

        return indices[:self.sample_frames:self.interval]

    def _get_whole_indices(self, data_info):
        num_frames = data_info['num_frames']
        if num_frames <= self.sample_frames:
            indices = list(range(1, num_frames+1, 1))
            while len(indices) < self.sample_frames:
                indices.extend(range(1, num_frames+1, 1))

            return indices[:self.sample_frames:self.interval]

        else:
            indices = list(range(1, num_frames+1, self.interval))

            return indices
    
    def _get_clips_indices(self, data_info, clips=10):
        num_frames = data_info['num_frames']
        indices = []

        clip_interval = 1
        if num_frames <= self.sample_frames + clip_interval*clips:
            
            extand_indices = list(range(1, num_frames+1, 1))
            while len(indices) < self.sample_frames + clip_interval*clips:
                extand_indices.extend(range(1, num_frames+1, 1))

            for i in range(clips):
                indices.extend(extand_indices[i : i+self.sample_frames : self.interval])
        else:
            clip_interval = int((num_frames - self.sample_frames) // clips)
            for i in range(clips):
                indices.extend(range(i+1+i*clip_interval, i+1+i*clip_interval+self.sample_frames, self.interval))
                
        return indices
            

    def _frames_loader(self, video_dir_path, frame_indices):
        video = []
        for i in frame_indices:
            if self.dataset == 'kinetics':
                image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            elif self.dataset == 'ucf101':
                image_path = os.path.join(video_dir_path, 'frame{:06d}.jpg'.format(i))
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    with Image.open(f) as img:
                        img = img.convert('RGB')
                        video.append(img)

            else:
                print(image_path)
                assert False, 'something error in frames path'

        return video

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.abspath('.'))
    # print(sys.path)

    train_set = Consecutive(dataset='ucf101', interval=2, train=False, test_mode='non_local')
    for i in range(len(train_set)):
        if train_set[i] != 320:
            print('error')


    class I3Dscale(object):
    # rescale piexls values in[0, 1] to [-1, 1]
        def __init__(self):
            return

        def __call__(self, data):
            return data*2 - 1.0

    import utils.transforms as ut_transforms
    import torchvision.transforms as T
    train_transforms = T.Compose([
    ut_transforms.GroupScale(256), # resize smaller edge to 256
    ut_transforms.GroupRandomCrop(224), # randomlly crop a 224x224 patch
    ut_transforms.GroupRandomHorizontalFlip(),
    ut_transforms.GroupStackToTensor(),
    # I3Dscale()
    ])

    train_set = Kinetics(transform=train_transforms)
    print('length of train dataset is :', len(train_set))
    print('first data size is :', train_set[0][0].size())
    print('first data is :', train_set[0][0])