import os

import numpy as np
import torchvision
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms as transforms
import random
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')

class deepfishDatasetSaveRAM(Dataset):
    """一个用于加载VOC数据集的自定义数据集"""
    def __init__(self, is_train, crop_size, image_dir,resizescale,numclass):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        self.resizescale=resizescale
        self.image_dir=image_dir
        self.featuresfname, self.labelsfname = self.read_deepfish_images(image_dir, resizescale,is_train=is_train)
        # self.features = [self.normalize_image(feature)
                         # for feature in self.filter(features)]
        # self.labels = self.filter(labels)
        self.num_classes=numclass
        # self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.featuresfname)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        # feature, label = processfeaturelabel(self.features[idx], self.labels[idx])
        featureimage = Image.open(os.path.join(os.path.join(self.image_dir, "temps"), self.featuresfname[idx] + ".jpg"))
        labelimage = Image.open(os.path.join(os.path.join(self.image_dir, "temps"), self.labelsfname[idx] + ".png"))
        feature = np.array(featureimage, np.float64)
        label = np.array(labelimage)

        feature=self.normalize_image(feature)

        return self.processfeaturelabel(feature, label)

    def __len__(self):
        return len(self.featuresfname)

    def read_deepfish_images(self,image_dir, resizescale,is_train=True,paint=False ,iszengyi=True):
        """读取所有VOC图像并标注"""
        txt_fname = os.path.join(image_dir, 'ImageSets', 'Segmentation',
                                 'train.txt' if is_train else 'val.txt')
        mode = torchvision.io.image.ImageReadMode.RGB
        with open(txt_fname, 'r') as f:
            images = f.read().split()
        featuresfname, labelsfname = [], []
        for i, fname in enumerate(images):

            featureimage = Image.open(os.path.join(os.path.join(image_dir, "JPEGImages"), fname + ".jpg"))
            labelimage = Image.open(os.path.join(os.path.join(image_dir, "SegmentationClass"), fname + ".png"))

            # resize
            w, h = featureimage.size

            featureimage = featureimage.resize((w//resizescale, h//resizescale), Image.BICUBIC)
            labelimage = labelimage.resize((w // resizescale, h // resizescale), Image.BICUBIC)
            self.wid = w//resizescale
            self.high = h// resizescale

            featureimage.save(os.path.join(os.path.join(image_dir, "temps"), fname + ".jpg"))
            labelimage.save(os.path.join(os.path.join(image_dir, "temps"), fname+'label' + ".png"))
            featuresfname.append(fname)
            labelsfname.append(fname+'label')

            # if paint:
            #     plt.figure()
            #     plt.imshow(features[-1])
            #     plt.show()

            #数据增益
            # if random.randint(0,10)>=4 and is_train and iszengyi:
            if random.randint(0, 10) >= 4 and iszengyi:
                newfeatureimage=copy.deepcopy(featureimage)
                newfeatureimage = transforms.RandomHorizontalFlip(p=1)(newfeatureimage)  # p表示概率
                newlabelimage = copy.deepcopy(labelimage)
                newlabelimage = transforms.RandomHorizontalFlip(p=1)(newlabelimage)  # p表示概率

                newfeatureimage.save(os.path.join(os.path.join(image_dir, "temps"), 'new'+fname + ".jpg"))
                newlabelimage.save(os.path.join(os.path.join(image_dir, "temps"), 'new'+fname + 'label' + ".png"))

                featuresfname.append('new'+fname)
                labelsfname.append('new'+fname+'label')
                # print('fip')
                # if paint:
                #     plt.figure()
                #     plt.imshow(features[-1])
                #     plt.show()

            # if random.randint(0, 10) >= 8 and iszengyi:
            #     newfeatureimage=copy.deepcopy(featureimage)
            #     newfeatureimage = transforms.RandomHorizontalFlip(p=1)(newfeatureimage)  # p表示概率
            #     newlabelimage = copy.deepcopy(labelimage)
            #     newlabelimage = transforms.RandomHorizontalFlip(p=1)(newlabelimage)  # p表示概率
            #
            #
            #     features.append(newfeatureimage)
            #     labels.append(newlabelimage)
            #     # print('fip')
            #     if paint:
            #         plt.figure()
            #         plt.imshow(features[-1])
            #         plt.show()

            # features.append(torchvision.io.read_image(os.path.join(
            #     image_dir, 'JPEGImages', f'{fname}.jpg')))
            # labels.append(torchvision.io.read_image(os.path.join(
            #     image_dir, 'SegmentationClass', f'{fname}.png'), mode))
        # for i in range(len(features)):
        #     features[i]=np.array(features[i], np.float64)
        #     labels[i]=np.array(labels[i])
        return featuresfname, labelsfname

    def processfeaturelabel(self,feature,label):
        # feature= np.transpose(np.array(feature, np.float64), [2,0,1])
        # label=np.array(label)
        feature = np.transpose(feature, [2, 0, 1])

        label[label >= self.num_classes] = self.num_classes
        # -------------------------------------------------------#
        #   转化成one_hot的形式,self.num_classes + 1是one_hot的长度
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.wid), int(self.high), self.num_classes + 1))

        return feature, label, seg_labels


# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = np.array(images)
    pngs        = np.array(pngs)
    seg_labels  = np.array(seg_labels)
    return images, pngs, seg_labels
