import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import src.config as config
from src.common.logger import get_logger
from src.data.utils import read_json, resize_image_wrt_asp

logger = get_logger(__name__)


class Layout:
    WIDTH = config.WIDTH
    HEIGHT = config.HEIGHT
    RESIZE_FACTOR = config.RESIZE_FACTOR

    def __init__(self, json_fn, png_fn):
        self.json_fn = json_fn
        self.png_fn = png_fn
        
        self.json_tree = read_json(json_fn)
        classes, bboxs = self.parse_bboxs_from_json_tree()
        self.gt_classes = list(classes)
        self.gt_bboxs = np.array(bboxs, dtype = np.float32)

        self.pred_bboxs = None
    
    @property
    def raw_image(self):
        img = Image.open(self.png_fn)
        new_img = resize_image_wrt_asp(img, factor = self.RESIZE_FACTOR)
        return new_img

    @property
    def rasterized_gt_bboxs(self):
        img = self.rasterize_bboxs(self.gt_bboxs, self.WIDTH, self.HEIGHT)
        new_img = resize_image_wrt_asp(img, factor = self.RESIZE_FACTOR)
        return new_img 

    @property
    def rasterized_pred_bboxs(self):
        if self.pred_bboxs is None:
            logger.warning('self.pred_bboxs is None, return None')
            return None
        img = self.rasterize_bboxs(self.pred_bboxs, self.WIDTH, self.HEIGHT)
        new_img = rasterize_bboxs(img, factor = self.RESIZE_FACTOR)
        return img

    @property
    def normalized_gt_bboxs(self):
        return self.normalize_bboxs(self.gt_bboxs)

    def __len__(self):
        return len(self.gt_bboxs)

    def normalize_bboxs(self, bboxs):
        normalized_bboxs = bboxs.copy()
        normalized_bboxs[:, [0, 2]] = normalized_bboxs[:, [0, 2]] / self.WIDTH
        normalized_bboxs[:, [1, 3]] = normalized_bboxs[:, [1, 3]] / self.HEIGHT
        return normalized_bboxs

    def unnormalize_bboxs(self, normalized_bboxs):
        bboxs = normalized_bboxs.copy()
        bboxs[:, [0, 2]] = bboxs[:, [0, 2]] * self.WIDTH
        bboxs[:, [1, 3]] = bboxs[:, [1, 3]] * self.HEIGHT
        return bboxs 

    def parse_bboxs_from_json_tree(self):
        layout_bboxs = []
        self._recursive_parse_json_tree(self.json_tree, 
                                   is_parse_parent = False, 
                                   bboxs = layout_bboxs)

        if len(layout_bboxs) == 0:
            classes, bboxs = [], []
        else:
            classes, bboxs = zip(*layout_bboxs)
        return classes, bboxs
    
    def _recursive_parse_json_tree(self, json_tree, is_parse_parent = False, bboxs = []):
        if 'children' in json_tree:
            if is_parse_parent:
                child_bbox = json_tree['bounds']
                child_class = self.parse_classs(json_tree['class'])
                bboxs.append((child_class, child_bbox))

            child_nodes = json_tree['children']
            for child_node in child_nodes:
                self._recursive_parse_json_tree(
                    child_node, is_parse_parent, bboxs = bboxs
                )

        else:
            child_bbox = json_tree['bounds']
            child_class = self.parse_class(json_tree['class'])
            bboxs.append((child_class, child_bbox))

    @staticmethod
    def parse_class(class_name):
        return class_name.split('.')[-1]

    @staticmethod
    def rasterize_bboxs(bboxs, width, height):
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img) 
        for bbox in bboxs:
            draw.rectangle(list(bbox), outline = 'red', fill = (0, 255, 0, 80))
        return img
