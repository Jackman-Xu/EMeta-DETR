from pathlib import Path
import torch
from torchvision.datasets.vision import VisionDataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
from io import BytesIO
import os
import os.path
from util.misc import get_local_rank, get_local_size

import datasets.transforms as T
# from datasets import (coco_base_class_ids, coco_novel_class_ids, \
#                       voc_base1_class_ids, voc_novel1_class_ids, \
#                       voc_base2_class_ids, voc_novel2_class_ids, \
#                       voc_base3_class_ids, voc_novel3_class_ids)


class SupportDataset(VisionDataset):
    """
    This SupportDataset is only used during inference stage.

    Support images used during training are sampled along with the query images for better speed.
    Therefore, to check support datasets used during training, please visit dataset.py and dataset_fewshot.py
    """
    def __init__(
            self,
            root,
            annFile,
            activatedClassIds,
            transforms=None,
            cache_mode=False,
            local_rank=0,
            local_size=1
    ) -> None:
        super(SupportDataset, self).__init__(root)
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        self.activatedClassIds = activatedClassIds
        self.classid2anno = {i: [] for i in self.activatedClassIds}
        self._transforms = transforms
        coco = COCO(annFile)
        for classid in self.activatedClassIds:
            annIds = coco.getAnnIds(catIds=classid)
            for annId in annIds:
                ann = coco.loadAnns(annId)[0]
                if 'area' in ann:
                    if ann['area'] < 5.0:
                        continue
                if 'ignore' in ann:
                    if ann['ignore']:
                        continue
                if 'iscrowd' in ann:
                    if ann['iscrowd'] == 1:
                        continue
                ann['image_path'] = coco.loadImgs(ann['image_id'])[0]['file_name']
                self.classid2anno[classid].append(ann) # classid2anno：{类别号：annos信息列表}
        self.len, self.classlendict = self.__calculate_length__() # maxlength，classlendict：{类别号：该annos的长度}
        self.prepare = ConvertCocoPolysToMask(return_masks=False)
        if self.cache_mode:
            self.cache = {}
            self.cache_images()

    def __getitem__(self, index: int):
        images = []
        targets = []
        class_ids = []

        # for classid in self.activatedClassIds: # 这里是每个指定的类别。首先是迭代index，例如index=1时，会对每一个类别中的下标为（1 % 类别图片数）的图片进行处理，然后加入列表
        #     i = index % self.classlendict[classid]
        #     target = self.classid2anno[classid][i]
            
        #     image_id = target['image_id']
        #     image_path = target['image_path']

        #     target = {'image_id': image_id, 'annotations': [target]}
        #     img_path = os.path.join(self.root, image_path)
            
        #     if self.cache_mode and (img_path in self.cache):
        #         img = Image.open(BytesIO(self.cache[img_path])).convert('RGB')
        #     else:
        #         img = Image.open(img_path).convert('RGB')
        #     img, target = self.prepare(img, target)
        #     if self._transforms is not None:
        #         original_target, original_img = target, img
        #         while True:
        #             img, target = self._transforms(original_img, original_target)
        #             # Make sure the object is not deleted after transforms, and it is not too small (mostly cut off)
        #             if target['boxes'].shape[0] == 1 and target['area'] >= original_target['area'] / 5.0:
        #                 break
        #     images.append(img)
        #     targets.append(target)
        #     class_ids.append(classid)



        for classid in self.activatedClassIds: # 这里是每个指定的类别。首先是迭代index，例如index=1时，会对每一个类别中的下标为（1 % 类别图片数）的图片进行处理，然后加入列表
            if index < self.classlendict[classid]:
                target = self.classid2anno[classid][index]
                
                image_id = target['image_id']
                image_path = target['image_path']

                target = {'image_id': image_id, 'annotations': [target]}
                img_path = os.path.join(self.root, image_path)

                if self.cache_mode and (img_path in self.cache):
                    img = Image.open(BytesIO(self.cache[img_path])).convert('RGB')
                else:
                    img = Image.open(img_path).convert('RGB')
                img, target = self.prepare(img, target)
                if self._transforms is not None:
                    original_target, original_img = target, img
                    while True:
                        img, target = self._transforms(original_img, original_target)
                        # Make sure the object is not deleted after transforms, and it is not too small (mostly cut off)
                        if target['boxes'].shape[0] == 1 and target['area'] >= original_target['area'] / 5.0:
                            break
            else:
                img = -1
                target = -1
            
    
            images.append(img)
            targets.append(target)
            class_ids.append(classid)
        return images, torch.as_tensor(class_ids), targets # 返回的有images的列表，对应指定的类别号列表，标注信息的列表（都是指定的类别）

    def __len__(self) -> int:
        return self.len

    def __calculate_length__(self):
        maxLength = 0
        lengthdict = dict()
        for k, v in self.classid2anno.items():
            cls_length = len(v)
            lengthdict[k] = cls_length
            if cls_length > maxLength:
                maxLength = len(v)
        return maxLength, lengthdict

    def cache_images(self):
        self.cache = {}
        for classid in self.activatedClassIds:
            for i in range(self.classlendict[classid]):
                if i % self.local_size != self.local_rank:
                    continue
                path = os.path.join(self.root, self.classid2anno[classid][i]['image_path'])
                if path in self.cache:
                    continue
                with open(os.path.join(path), 'rb') as f:
                    self.cache[path] = f.read()


def make_support_transforms():
    """
    Transforms for support images during inference stage.

    For transforms of support images during training, please visit dataset.py and dataset_fewshot.py
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704]

    return T.Compose([ # 在inference阶段下，support数据的transform方式不再进行随机裁剪和颜色扰动，而是使用原图
        T.RandomHorizontalFlip(), # 随机水平翻转
        T.RandomResize(scales, max_size=768),
        normalize,
    ])


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def build_val_support_dataset(args, img_folder, ann_file, activatedClassIds):
    return SupportDataset(img_folder, ann_file,
                            activatedClassIds=activatedClassIds,
                            transforms=make_support_transforms(),
                            cache_mode=args.cache_mode,
                            local_rank=get_local_rank(),
                            local_size=get_local_size())
