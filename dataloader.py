import torch
from torch.utils.data import Dataset
from PIL import Image
import glob, os, copy, random


class SmoothData(Dataset):
    def __init__(self, path, split_ratio={'train': 0.65, 'val': 0.1, 'test': 0.25}, transform=None):
        self.transforms = transform
        self.category = {}
        self.name = {'input', 'target_SPN', 'target_TPN', 'target_TSAFN'}
        self.mode = None
        self.modeset = ['train', 'val', 'test']

        check_folder = set()
        for directory, folders, _ in os.walk(path):
            for f in folders:
                self.category[f] = glob.glob(os.path.join(directory, f) + '\\*')
                check_folder.add(f)
            break  # only retrieve one level

        assert not (self.name - check_folder), 'folder name must be same as declared in self.name'
        assert sum(split_ratio.values()) == 1, 'the summation of values in split_ratio must be one'

        self.total = len(self.category['input'])

        index_shuffle = list(range(self.total))
        random.shuffle(index_shuffle)

        train_amount = int(self.total * split_ratio['train'])
        val_amount = int(self.total * split_ratio['val'])
        self.amount = {'train': train_amount,
                       'val': val_amount,
                       'test': self.total - train_amount - val_amount}  # must use minus, otherwise not full.

        index_train = index_shuffle[: self.amount['train']]
        index_val = index_shuffle[self.amount['train']: self.amount['train'] + self.amount['val']]
        index_test = index_shuffle[self.amount['train'] + self.amount['val']:]

        # the file retrieval order is not sequentially from 1 to end, but may be like 1, 10, 11, ... , 2, 20 ..
        # this part is to align the image index into sequential order for target_SPN and target_TSAFN
        target_length = len(self.category['target_SPN'])
        temp1 = [0 for i in range(target_length)]
        temp2 = copy.deepcopy(temp1)
        for i in range(target_length):
            img_path = self.category['target_SPN'][i]
            _, img_name = os.path.split(img_path)
            img_index = int(img_name.split('.')[0])
            temp1[img_index] = img_path

            img_path = self.category['target_TSAFN'][i]
            _, img_name = os.path.split(img_path)
            img_index = int(img_name.split('.')[0])
            temp2[img_index] = img_path
        self.category['target_SPN'] = temp1
        self.category['target_TSAFN'] = temp2

        self.collection = {'train': ([[0 for i in range(4)] for i in range(self.amount['train'])], index_train),
                           'val': ([[0 for i in range(4)] for i in range(self.amount['val'])], index_val),
                           'test': ([[0 for i in range(4)] for i in range(self.amount['test'])], index_test)}

        for m in self.modeset:
            for subset_index, alldata_index in enumerate(self.collection[m][1]):
                _, img_name = os.path.split(self.category['input'][alldata_index])
                img_index = int(img_name.split('_')[0])

                self.collection[m][0][subset_index] = [self.category['input'][alldata_index],
                                                       self.category['target_TPN'][alldata_index],
                                                       self.category['target_SPN'][img_index],
                                                       self.category['target_TSAFN'][img_index]]

    def __call__(self, mode):
        assert mode in self.modeset, 'mode must be either train, val or test'
        self.mode = mode
        return self

    def __len__(self):
        if self.mode:
            return self.amount[self.mode]
        else:
            raise Exception(
                'Firstly, you must use self.setmode(mode) to set which data set (train, val, test) to be use.')

    def __getitem__(self, i):
        # check if mode is set
        len(self)

        dataset = self.collection[self.mode][0]
        inpu = Image.open(dataset[i][0])
        target_TPN = Image.open(dataset[i][1])
        target_SPN = Image.open(dataset[i][2])
        target_TSAFN = Image.open(dataset[i][3])

        if self.transforms:
            inpu = self.transforms(inpu)
            target_TPN = self.transforms(target_TPN)
            target_SPN = self.transforms(target_SPN)
            target_TSAFN = self.transforms(target_TSAFN)

        return {'input': inpu, 'TPN': target_TPN, 'SPN': target_SPN, 'TSAFN': target_TSAFN}