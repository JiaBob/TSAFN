import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob, os, copy, random
from torchvision import transforms


class SplitData:
    def __init__(self, path, index=None, unknown=False, split_ratio={'train': 0.65, 'val': 0.1, 'test': 0.25}):
        self.category = {}
        self.name = {'input', 'target_spn', 'target_tpn', 'target_tsafn'}
        self.modeset = ['train', 'val', 'test']
        self.unknown = unknown

        if self.unknown:
            img_path = glob.glob(path + '\\*')
            self.collection = {'unknown': img_path}
        else:
            check_folder = set()
            for directory, folders, _ in os.walk(path):
                for f in folders:
                    print(os.path.join(directory, f) + '\\*')
                    self.category[f] = glob.glob(os.path.join(directory, f) + '\\*')
                    check_folder.add(f)
                break  # only retrieve one level

            assert not (self.name - check_folder), 'folder name must be same as declared in self.name'
            assert sum(split_ratio.values()) == 1, 'the summation of values in split_ratio must be one'

            self.total = len(self.category['input'])
            print(self.category)
            index_shuffle = list(range(self.total))
            random.shuffle(index_shuffle)

            train_amount = int(self.total * split_ratio['train'])
            val_amount = int(self.total * split_ratio['val'])
            self.amount = {'train': train_amount,
                           'val': val_amount,
                           'test': self.total - train_amount - val_amount}  # must use minus, otherwise not full.
            if index:
                index_train = index['train']
                index_val = index['val']
                index_test = index['test']
            else:
                index_train = index_shuffle[: self.amount['train']]
                index_val = index_shuffle[self.amount['train']: self.amount['train'] + self.amount['val']]
                index_test = index_shuffle[self.amount['train'] + self.amount['val']:]
            self.index_dict = {'train': index_train, 'val': index_val, 'test': index_test}

            # the file retrieval order is not sequentially from 1 to end, but may be like 1, 10, 11, ... , 2, 20 ..
            # this part is to align the image index into sequential order for target_SPN and target_TSAFN
            target_length = len(self.category['target_spn'])
            temp1 = [0 for i in range(target_length)]
            temp2 = copy.deepcopy(temp1)
            for i in range(target_length):
                img_path = self.category['target_spn'][i]
                _, img_name = os.path.split(img_path)
                img_index = int(img_name.split('.')[0])
                temp1[img_index] = img_path

                img_path = self.category['target_tsafn'][i]
                _, img_name = os.path.split(img_path)
                img_index = int(img_name.split('.')[0])
                temp2[img_index] = img_path

            self.category['target_spn'] = temp1
            self.category['target_tsafn'] = temp2

            self.collection = {'train': [[0 for i in range(4)] for i in range(self.amount['train'])],
                               'val': [[0 for i in range(4)] for i in range(self.amount['val'])],
                               'test': [[0 for i in range(4)] for i in range(self.amount['test'])]}

            for m in self.modeset:
                for subset_index, alldata_index in enumerate(self.index_dict[m]):
                    _, img_name = os.path.split(self.category['input'][alldata_index])
                    img_index = int(img_name.split('_')[0])
                    self.collection[m][subset_index] = [self.category['input'][alldata_index],
                                                        self.category['target_tpn'][alldata_index],
                                                        self.category['target_spn'][img_index],
                                                        self.category['target_tsafn'][img_index]]


    def __call__(self, mode, random_crop_size=None):
        return SmoothData(self.collection[mode], mode, random_crop_size)


class SmoothData(Dataset):
    def __init__(self, dataset, mode='train', random_crop_size=None):
        self.dataset = dataset
        self.mode = mode
        self.random_crop_size = random_crop_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        if self.mode == 'unknown':
            inpu = Image.open(self.dataset[i])
            rescale_size = (512, 512)

            inpu = transforms.functional.resize(inpu, rescale_size)
            inpu = transforms.ToTensor()(inpu)
            return inpu

        else:
            inpu = Image.open(self.dataset[i][0])
            target_tpn = Image.open(self.dataset[i][1])
            target_spn = Image.open(self.dataset[i][2])
            target_tsafn = Image.open(self.dataset[i][3])

            rescale_size = (400, 400)
            inpu = transforms.functional.resize(inpu, rescale_size)
            target_spn = transforms.functional.resize(target_spn, rescale_size)
            target_tpn = transforms.functional.resize(target_tpn, rescale_size)
            target_tsafn = transforms.functional.resize(target_tsafn, rescale_size)

            # only do random crop on training set and when random_crop_size is specified.
            if self.mode == 'train' and self.random_crop_size:

                x, y = self.getCrop(rescale_size, self.random_crop_size)

                inpu = transforms.functional.crop(inpu, x, y, *self.random_crop_size)
                target_spn = transforms.functional.crop(target_spn, x, y, *self.random_crop_size)
                target_tpn = transforms.functional.crop(target_tpn, x, y, *self.random_crop_size)
                target_tsafn = transforms.functional.crop(target_tsafn, x, y, *self.random_crop_size)

            inpu = transforms.ToTensor()(inpu)
            target_tpn = transforms.ToTensor()(target_tpn)
            target_spn = transforms.ToTensor()(target_spn)
            target_tsafn = transforms.ToTensor()(target_tsafn)

            return {'input': inpu, 'TPN': target_tpn, 'SPN': target_spn, 'TSAFN': target_tsafn}

    def getCrop(self, size, crop_size):
        cropH, cropW = crop_size
        h, w = size
        y = np.random.randint(h - cropH)
        x = np.random.randint(w - cropH)
        return x, y

