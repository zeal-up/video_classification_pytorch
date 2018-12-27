import torch.utils.data as data 

from PIL import Image
import os
import numpy as np 

def read_list(root, split, train):
    trainlist_path = os.path.join(root, 'train_val_split', 'trainlist0'+str(split)+'.txt')
    testlist_path = os.path.join(root, 'train_val_split', 'testlist0'+str(split)+'.txt')
    train_list = []
    test_list = []
    if train:
        with open(trainlist_path, 'r') as f:
            for line in f:
                list = line.strip().split(' ')
                list[1], list[2] = int(list[1]), int(list[2])
                train_list.append(list)
        return train_list
    else:
        with open(testlist_path, 'r') as f:
            for line in f:
                list = line.strip().split(' ')
                list[1], list[2] = int(list[1]), int(list[2])
                test_list.append(list)
        return test_list


class TSNDataSet(data.Dataset):
    '''
    root->
            UCF101->
                    jpegs_256->
                            v_ApplyEyeMakeup_g01_c01->
                                    frame000001.jpg
                                    frame000002.jpg
                                        ....

                            v_ApplyEyeMakeup_g01_c02->
                                        ....
                                ....
                    tvl1_flow->
                            u->
                                v_ApplyEyeMakeup_g01_c01->
                                    frame000001.jpg
                                    frame000002.jpg
                                        ...

                                v_ApplyEyeMakeup_g01_c02->
                                    ....
                            v->
                                v_ApplyEyeMakeup_g01_c01->
                                    frame000001.jpg
                                    frame000002.jpg
                                        ...

                                v_ApplyEyeMakeup_g01_c02
                                    ....

                    train_val_split->
                            testlist01.txt  
                            testlist02.txt  
                            testlist03.txt  
                            trainlist01.txt  
                            trainlist02.txt  
                            trainlist03.txt
            HMDB51->
    '''
    def __init__(self, root='./videos_dataset', num_segments=3, dataset='UCF101', new_length=1, 
                modality='RGB', transform=None, random_shift=True, split=1, train=True):
        super().__init__()

        self.root = os.path.join(root, dataset)
        self.num_segments = num_segments
        self.new_length = new_length
        self.transform = transform
        self.random_shift = random_shift
        self.train = train
        self.modality = modality
        self.video_list = read_list(self.root, split, train)
        # train_data = {[path, num_frames, label], [...], ...}
        # test_data = {[path, num_frames]}

        # store num_class
        if dataset == 'UCF101':
            self.num_class = 101
        elif dataset == 'HMDB51':
            self.num_class = 51
        elif dataset == 'KINETICS':
            self.num_class = 400
        else:
            raise ValueError('Unknown dataset ' + dataset)
    
    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.train:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        class_name = record[0]
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(class_name, p)
                images.extend(seg_imgs)
                if p < record[1]:
                    p += 1

        process_data = self.transform(images)
        return process_data, record[2]

    def _load_image(self, class_name, idx):
        if self.modality == 'RGB':
            return [Image.open(os.path.join(self.root, 'jpegs_256', class_name, 'frame{:06d}.jpg'.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(self.root, 'tvl1_flow/u', class_name, 'frame{:06d}.jpg'.format(idx))).convert('L')
            y_img = Image.open(os.path.join(self.root, 'tvl1_flow/v', class_name, 'frame{:06d}.jpg'.format(idx))).convert('L')

            return [x_img, y_img]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        
        num_frames = record[1]

        average_duration = (num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration, size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(np.random.randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        num_frames = record[1]
        if num_frames > self.num_segments + self.new_length - 1:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1


    def _get_test_indices(self, record):
        num_frames = record[1]

        tick = (num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    trainset = TSNDataSet(train=True)
    testset = TSNDataSet(train=False)

    print(len(trainset))
    print(len(testset))