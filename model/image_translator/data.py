
import os.path
from PIL import Image
from skimage import exposure, io
import torch.utils.data as data
import numpy as np


def default_loader(path):
    try:
        image = Image.open(path).convert('RGB')
    except:
        image = path.replace('sim0', '')
        image = image.replace('sim1', '')
        image = image.replace('sim2', '')
        image = Image.open(path).convert('RGB')
    return image

def transform_loader(path):
    image = io.imread(path)
    image = exposure.equalize_adapthist(image, clip_limit=0.1) #CLAHE pre-processing
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image


def default_filelist_reader(filelist):
    im_list = []
    with open(filelist, 'r') as rf:
        for line in rf.readlines():
            im_path = line.strip()
            im_list.append(im_path)
    return im_list


class ImageLabelFilelist(data.Dataset):
    def __init__(self,
                 root,
                 filelist,
                 transform=None,
                 filelist_reader=default_filelist_reader,
                 loader=default_loader,
                 dataset='animals',
                 return_paths=False):
        self.root = root
        self.im_list = filelist_reader(os.path.join(filelist))
        self.transform = transform
        if dataset in ['animals']:
            self.loader = loader
        else:
            exit('dataset not found')
        labels = [path.split('/')[0] for path in self.im_list]
        self.classes = sorted(list(set(labels)))
        self.class_to_idx = {self.classes[i]: i for i in
                        range(len(self.classes))}
        k = -1
        seen_class_names = []
        str_to_idx = {}
        for i in range(len(labels)):
            if labels[i] not in seen_class_names:
                seen_class_names.append(labels[i])
                k += 1
            str_to_idx[labels[i]] = k
            
        self.labels = [str_to_idx[str_label] for str_label in labels]
        if dataset == 'cub':
            self.imgs = [(im_path.split(',')[0], self.class_to_idx[im_path.split(',')[1]]) for
                        im_path in self.im_list]
        elif dataset == 'miniImagenet':
            self.imgs = [(im_path, \
                        self.class_to_idx[im_path[:len('n01930112')]]) for
                        im_path in self.im_list]
        else:
            self.imgs = [(im_path, self.class_to_idx[im_path.split('/')[0]]) for
                        im_path in self.im_list]
        self.return_paths = return_paths

    def __getitem__(self, index):
        im_path, label = self.imgs[index]
        path = os.path.join(self.root, im_path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, label, path
        else:
            return img, label

    def __len__(self):
        return len(self.imgs)
