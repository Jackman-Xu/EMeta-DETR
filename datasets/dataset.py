import os
import random
from PIL import Image
import torch
import torch.utils.data
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class DetectionDataset(TvCocoDetection):
    def __init__(self, args, img_folder, ann_file, transforms, support_transforms, return_masks, activated_class_ids,
                 with_support=True, cache_mode=False, local_rank=0, local_size=1):
        super(DetectionDataset, self).__init__(img_folder, ann_file, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.with_support = with_support
        self.activated_class_ids = activated_class_ids
        self._transforms = transforms # 针对query的transforms
        self.prepare = ConvertCocoPolysToMask(return_masks) # prepare为过滤重叠物体，并且将bbox坐标格式转化为xyxy（0～w，0～h）模式。
        """
        If with_support = True, this dataset will also produce support images and support targets.
        with_support should be set to True for training, and should be set to False for inference.
          * During training, support images are sampled along with query images in this dataset.
          * During inference, support images are sampled from dataset_support.py
        """
        if self.with_support:
            self.NUM_SUPP = args.total_num_support
            self.NUM_MAX_POS_SUPP = args.max_pos_support
            self.support_transforms = support_transforms
            self.build_support_dataset(ann_file)



    def __getitem__(self, idx): # 依次获取每张图片，以及图片内的所有标注信息，每张图片可能有包含指定类别的信息，也有可能一个指定类别信息都不包含
        img, target = super(DetectionDataset, self).__getitem__(idx) # img为加载的经过transform的图片数据，target为图片对应的annos信息
        target = [anno for anno in target if anno['category_id'] in self.activated_class_ids] # 过滤掉该图片的非指定的类别的annos信息，仅保留相应类别的annos信息
        image_id = self.ids[idx]

        target = {'image_id': image_id, 'annotations': target} # 此时target里面的bbox信息已经是x1,y1,w,h（x1为0～w，y1为0～h）了（绝对坐标）
        img, target = self.prepare(img, target) # img与target，target：{image_id, bbox, label, area等}，bbox变为x1,y1,x2,y2（绝对坐标）
        if self._transforms is not None:
            img, target = self._transforms(img, target) # 用于训练的1张Query，图片内符合指定类别条件的全部target数据，这里的target的bbox形式为xc,yc,w,h （相对坐标0～1）
        
        if self.with_support: # 构造和该query相匹配的support数据
            support_images, support_class_ids, support_targets = self.sample_support_samples(target) # 生成这张Query对应的Supoort图片，以及相应的标注信息
            return img, target, support_images, support_class_ids, support_targets # 返回当前迭代到的Query图片，以及其对应的各指定类别与随机的非指定类别的1张Support图像
        else:
            return img, target



    def build_support_dataset(self, ann_file): # 对数据集，依据指定的类别，初步完成过滤（面积不过小，无ignore，单独个体），得到制定类别下符合条件的各个标注数据
        self.classid2anno = {i: [] for i in self.activated_class_ids} # 操作仅与Support集有关，与Query无关
        coco = COCO(ann_file)
        for classid in self.activated_class_ids: # 遍历每一个指定激活的类
            annIds = coco.getAnnIds(catIds=classid) # 获取当前指定类所对应的所有标注信息的annIDs（即标注信息的id号）
            for annId in annIds:
                ann = coco.loadAnns(annId)[0] # 读取对应annId号的完整标注信息
                if 'area' in ann:
                    if ann['area'] < 5.0: # 面积过小的不要
                        continue
                if 'ignore' in ann: # 有ignore的不要
                    if ann['ignore']:
                        continue
                if 'iscrowd' in ann: # 如果图片内有重叠物体的不要，只要单独的个体
                    if ann['iscrowd'] == 1:
                        continue
                ann['image_path'] = coco.loadImgs(ann['image_id'])[0]['file_name'] # 值例如'000000122851.jpg'
                self.classid2anno[classid].append(ann) 


    def sample_support_samples(self, target): # 这里的target指的是1张Query图片里面的target信息，包括bbox以及label
        positive_labels = target['labels'].unique() # 这张图片中含有的各种positive label，即图片中涉及到的类别，就是positive label
        num_positive_labels = positive_labels.shape[0] # 这张图片中的positive label的数量
        positive_labels_list = positive_labels.tolist() # 指定的类别列表
        negative_labels_list = list(set(self.activated_class_ids) - set(positive_labels_list)) # 未指定的类别列表

        # Positive labels in a batch < TRAIN_NUM_POSITIVE_SUPP: we include additional labels as negative samples
        if num_positive_labels <= self.NUM_MAX_POS_SUPP: # 如果这张图里面的类别数量 <= 规定的support分支能够共存的最大类别数，则使用negative label进行填充，填充到total_num_support大小（15）
            sampled_labels_list = positive_labels_list
            sampled_labels_list += random.sample(negative_labels_list, k=self.NUM_SUPP - num_positive_labels)
        # Positive labels in a batch > TRAIN_NUM_POSITIVE_SUPP: remove some positive labels.
        else: # 如果这张图里面的类别数量 > 规定的support分支能够共存的最大类别数，则随机在positive label中进行个数采样
            sampled_positive_labels_list = random.sample(positive_labels_list, k=self.NUM_MAX_POS_SUPP)
            sampled_negative_labels_list = random.sample(negative_labels_list, k=self.NUM_SUPP - self.NUM_MAX_POS_SUPP) # 然后再从negative label中随机采样补齐后面的5个（total_num_support（15）- max_pos_support（10））
            sampled_labels_list = sampled_positive_labels_list + sampled_negative_labels_list
            # -----------------------------------------------------------------------
            # NOTE: There is no need to filter gt info at this stage.
            #       Filtering is done when formulating the episodes.
            # -----------------------------------------------------------------------
        # ⬆️ 完成的是构建一个总数量为 total_num_support（15）的label list，而不是image list。
        # 并且说明这个构造的label list中除了positive label外，一定也会有negative label
        
        support_images = []
        support_targets = []
        support_class_ids = []
        for class_id in sampled_labels_list: # len（sampled_labels_list）= 15
            i = random.randint(0, len(self.classid2anno[class_id]) - 1)
            support_target = self.classid2anno[class_id][i] # 随机抽到的sample_labels_list中的一个类别，在这个类别中的标注信息随机选择一条target
            
            image_id = support_target['image_id']
            image_path = support_target['image_path']

            support_target = {'image_id': image_id, 'annotations': [support_target]} 
            support_image_path = os.path.join(self.root, image_path)

            support_image = Image.open(support_image_path).convert('RGB')
            support_image, support_target = self.prepare(support_image, support_target) # img与target，target：{image_id, bbox, label, area等}
            if self.support_transforms is not None:
                org_support_target, org_support_image = support_target, support_image
                while True:
                    support_image, support_target = self.support_transforms(org_support_image, org_support_target)
                    # Make sure the object is not deleted after transforms, and it is not too small (mostly cut off)
                    if support_target['boxes'].shape[0] == 1 and support_target['area'] >= org_support_target['area'] / 5.0:
                        break
            support_images.append(support_image) # 相当于是Query中的每一个positive support类别，仅带有一张图片，不重复随机采样的negative label，对应的也仅一张图片
            support_targets.append(support_target)
            support_class_ids.append(class_id)
        return support_images, torch.as_tensor(support_class_ids), support_targets


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


class ConvertCocoPolysToMask(object): # 数据格式转换，过滤重叠的物体，并且修正坐标为xyxy模式（0～w，0～h）。
    def __init__(self, return_masks=False):
        self.return_masks = return_masks # 这个masks操作使用在实例分割任务的

    def __call__(self, image, target): # image：Image Open过的图片，target：{'image_id': image_id, 'annotations': target}
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0] # 该图片中对应的类别的标注信息是不密集的物体

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2] # boxes变为[x1,y1,x2,y2]
        boxes[:, 0::2].clamp_(min=0, max=w) # 修正坐标值，范围为0～w
        boxes[:, 1::2].clamp_(min=0, max=h) # 修正坐标值，范围为0～h

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




def make_transforms(image_set):
    """
    Transforms for query images during the base training phase or fewshot finetune phase.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set in ['base_train', 'fewshot_finetune']:
        return T.Compose([ # 这些T.函数的操作，会自动对bbox的坐标连同图像一起处理
            T.RandomHorizontalFlip(), # 随机水平翻转
            T.RandomColorJitter(p=0.3333), # 随机颜色扰动
            T.RandomSelect( # 随机选择其中1个操作（二选一）
                T.RandomResize(scales, max_size=1152),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1152),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1152),
            normalize,
        ])
    
    raise ValueError(f'unknown {image_set}')


def make_support_transforms():
    """
    Transforms for support images during the base training phase or fewshot finetune phase.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672]

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomColorJitter(p=0.25),
        T.RandomSelect(
            T.RandomResize(scales, max_size=672),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=672),
            ])
        ),
        normalize,
    ])


def build_train(args, img_folder, ann_file, image_set, activated_class_ids, with_support):
    return DetectionDataset(args, img_folder, ann_file,
                            transforms=make_transforms(image_set),
                            support_transforms=make_support_transforms(),
                            return_masks=False,
                            activated_class_ids=activated_class_ids,
                            with_support=with_support,
                            cache_mode=args.cache_mode,
                            local_rank=get_local_rank(),
                            local_size=get_local_size())