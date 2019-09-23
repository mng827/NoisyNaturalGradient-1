from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.transforms as transforms
import torchvision
import torch

import numpy as np
import nibabel as nib
import os
from misc import image_utils
import csv


class Flatten(object):
    def __call__(self, tensor):
        return tensor.view(-1)

    def __repr__(self):
        return self.__class__.__name__


class Transpose(object):
    def __call__(self, tensor):
        return tensor.permute(1, 2, 0)

    def __repr__(self):
        return self.__class__.__name__


def load_pytorch(config):
    if config.dataset == 'cifar10':
        if config.data_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Transpose()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Transpose()
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Transpose()
        ])
        trainset = torchvision.datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=test_transform)
    elif config.dataset == 'cifar100':
        if config.data_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                Transpose()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                Transpose()
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            Transpose()
        ])
        trainset = torchvision.datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=test_transform)
    elif config.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            Flatten(),
        ])
        trainset = torchvision.datasets.MNIST(root=config.data_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=config.data_path, train=False, download=True, transform=transform)
    elif config.dataset == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            Flatten(),
        ])
        trainset = torchvision.datasets.FashionMNIST(root=config.data_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=config.data_path, train=False, download=True, transform=transform)

    elif config.dataset == 'ukbb':
        trainset = VolumetricImageDataset(data_root_dir=config.train_data_path,
                                          data=get_image_list(config.train_data_file),
                                          image_size=config.image_size,
                                          network_type=config.network_type,
                                          num_images_limit=config.num_images_limit,
                                          augment=config.data_aug,
                                          shift=config.data_aug_shift,
                                          rotate=config.data_aug_rotate,
                                          scale=config.data_aug_scale)

        testset = VolumetricImageDataset(data_root_dir=config.validation_data_path,
                                          data=get_image_list(config.validation_data_file),
                                          image_size=config.image_size,
                                          network_type=config.network_type,
                                          num_images_limit=config.num_images_limit,
                                          augment=False)

        return torch.utils.data.DataLoader(trainset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           num_workers=config.num_workers,
                                           collate_fn=concatenate_samples), \
               torch.utils.data.DataLoader(testset,
                                           batch_size=config.test_batch_size,
                                           shuffle=False,
                                           num_workers=config.num_workers,
                                           collate_fn=concatenate_samples)

    else:
        raise ValueError("Unsupported dataset!")

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config.test_batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers)
    return trainloader, testloader


def concatenate_samples(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    info = batch[0][2]
    whole_image = [item[3] for item in batch]
    whole_label = [item[4] for item in batch]

    return [np.concatenate(data, axis=0), np.concatenate(label, axis=0), info, whole_image, whole_label]


class VolumetricImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_root_dir, data, image_size, network_type, num_images_limit, augment, shift=10, rotate=10,
                 scale=0.1):
        self.data_root_dir = data_root_dir
        self.data = data
        self.image_size = image_size
        self.network_type = network_type
        self.num_images_limit = num_images_limit
        self.augment = augment
        self.shift = shift
        self.rotate = rotate
        self.scale = scale

    def __getitem__(self, index):
        image_name = self.data['image_filenames'][index]
        label_name = self.data['label_filenames'][index]
        phase_number = self.data['phase_numbers'][index]

        image_id = os.path.basename(image_name).split('.')[0]
        info = {}
        info['Name'] = image_id
        info['PhaseNumber'] = phase_number

        nib_image = nib.load(os.path.join(self.data_root_dir, image_name))
        info['PixelResolution'] = nib_image.header['pixdim'][1:4]
        info['ImageSize'] = nib_image.header['dim'][1:4]
        info['AffineTransform'] = nib_image.header.get_best_affine()
        info['Header'] = nib_image.header.copy()

        whole_image = nib_image.get_data()
        whole_label = nib.load(os.path.join(self.data_root_dir, label_name)).get_data()

        if np.ndim(whole_image) == 4:
            whole_image = whole_image[:, :, :, phase_number]
            whole_label = whole_label[:, :, :, phase_number]

        # # For ACDC
        # whole_label = 4 - whole_label
        # whole_label[whole_label == 4] = 0

        whole_image_orig = np.copy(whole_image)

        clip_min = np.percentile(whole_image, 1)
        clip_max = np.percentile(whole_image, 99)
        whole_image = np.clip(whole_image, clip_min, clip_max)
        whole_image = (whole_image - whole_image.min()) / float(whole_image.max() - whole_image.min())

        # whole_image = (whole_image - np.mean(whole_image, dtype=np.float32)) / np.std(whole_image, dtype=np.float32)

        # Pad image into square and resize here
        # whole_image = image_utils.zero_pad(whole_image)
        # whole_label = image_utils.zero_pad(whole_label)
        #
        # whole_image = image_utils.resize_image(whole_image, [image_size, image_size], interpolation_order=1)
        # whole_label = image_utils.resize_image(whole_label, [image_size, image_size], interpolation_order=0) * 255

        x, y, z = whole_image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        cropped_image = image_utils.crop_image(whole_image, x_centre, y_centre, [self.image_size, self.image_size])
        cropped_label = image_utils.crop_image(whole_label, x_centre, y_centre, [self.image_size, self.image_size])

        # Perform data augmentation
        if self.augment:
            if self.network_type == '2d' or self.network_type == 'bayesian2d' or self.network_type == 'bayesian2d_mc_dropout':
                cropped_image, cropped_label = image_utils.augment_data_2d(cropped_image, cropped_label,
                                                                           preserve_across_slices=False,
                                                                           max_shift=self.shift, max_rotate=self.rotate,
                                                                           max_scale=self.scale)
            elif self.network_type == '2.5d':
                cropped_image, cropped_label = image_utils.augment_data_2d(cropped_image, cropped_label,
                                                                           preserve_across_slices=True,
                                                                           max_shift=self.shift, max_rotate=self.rotate,
                                                                           max_scale=self.scale)
            else:
                raise Exception("Unknown type in data augmentation.")

        if self.network_type == '2d' or self.network_type == 'bayesian2d' or self.network_type == 'bayesian2d_mc_dropout':
            # Put into NHWC format
            batch_images = np.expand_dims(np.transpose(cropped_image, axes=(2, 0, 1)), axis=-1)
            batch_labels = np.expand_dims(np.transpose(cropped_label, axes=(2, 0, 1)), axis=-1)

            if batch_images.shape[0] > self.num_images_limit:
                slices = sorted(list(np.random.permutation(batch_images.shape[0]))[:self.num_images_limit])
                batch_images = batch_images[slices, :, :, :]
                batch_labels = batch_labels[slices, :, :, :]
                # print("Warning: Number of slices limited to {} to fit GPU".format(num_images_limit))

        elif self.network_type == '2.5d':
            # Put into NDHWC format
            batch_images = np.expand_dims(np.transpose(cropped_image, axes=(0, 3, 1, 2)), axis=-1)
            batch_labels = np.expand_dims(np.transpose(cropped_label, axes=(0, 3, 1, 2)), axis=-1)

        else:
            raise Exception("Unknown type in get_batch.")

        return batch_images, batch_labels, info, whole_image_orig, whole_label
        # return batch_images, batch_labels

    def __len__(self):
        return len(self.data['image_filenames'])


def get_image_list(csv_file):
    image_list, label_list, phase_list = [], [], []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            phase_numbers = eval(row['phase_numbers'])

            for phase in phase_numbers:
                image_list.append(row['image_filenames'].strip())
                label_list.append(row['label_filenames'].strip())
                phase_list.append(phase)

    data_list = {}
    data_list['image_filenames'] = image_list
    data_list['label_filenames'] = label_list
    data_list['phase_numbers'] = phase_list

    return data_list
