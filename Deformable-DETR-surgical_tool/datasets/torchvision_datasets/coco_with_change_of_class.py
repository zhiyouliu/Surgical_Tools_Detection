# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""

"""
This file is modified by Steven Kan. We mainly change the output classes with
only the objects from the class "people" (1 -> 1), "vehicle" (3,6,8->2) and "traffic light" (10 -> 3).
"""
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        target = self.modify_target(target)  #added by Steven Kan
        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def modify_target(self, target):
        """
        This function is added by Steven Kan. We will only keep the objects belonging to
        class of "people" (1), "vehicle" (3,6,8) and "traffic light" (10). Then, we will change the output
        class label into "people" (1 -> 1), "vehicle" (3,6,8->2) and "traffic light" (10 -> 3).
        
        Input:
            target: the target from COCO dataset. It contains a list of dictionary, where each single dictionary
                    is for a single object on the image. Therefore, we will find the objects that we want through
                    the "category_id" feature in each dictionary adn decide the dictionary we want to keep.
        """
        #print("len of original target: ", len(target))

        coco_to_carla_class = {1: 1,
                               3: 2,
                               6: 2,
                               8: 2,
                               10: 3}

        target_keep = []
        for i in range(len(target)):
            curr_target = target[i]
            label = curr_target["category_id"]
            if label in coco_to_carla_class:
                new_label = coco_to_carla_class[label]
                curr_target["category_id"] = new_label
                target_keep.append(curr_target)
            else:
                continue
            
        #print("len of labels keep: ", len(target_keep))
        return target_keep

