import os
import sys
import cv2
import json
import copy
import numpy as np
from utils import (seg_class_description_dict,
                                                seg_class_easy_name_dict,
                                                seg_class_shape_description_dict,
                                                seg_class_easy_description_dict)
from utils import get_file_list


class PrepareData(object):
    def __init__(self,
                 image_dir,
                 seg_dir,
                 save_image_dir,
                 save_mask_dir,
                 save_image_w_mask_dir,
                 json_path=None):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.save_image_dir = save_image_dir
        self.save_mask_dir = save_mask_dir
        self.save_image_w_mask_dir = save_image_w_mask_dir
        self.json_path = json_path
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)
        if not os.path.exists(save_mask_dir):
            os.makedirs(save_mask_dir)
        if not os.path.exists(save_image_w_mask_dir):
            os.makedirs(save_image_w_mask_dir)

        if 'binary_masks' in seg_dir:
            self.seg_class_list = ['Background', 'Instrument']
        elif 'parts_masks' in seg_dir:
            self.seg_class_list = ['Background', 'Shaft', 'Wrist', 'Claspers']
        elif 'instruments_masks' in seg_dir:
            self.seg_class_list = ['Background', 'Bipolar Forceps', 'Prograsp Forceps', 'Large Needle Driver',
                                   'Vessel Sealer', 'Grasping Retractor', 'Monopolar Curved Scissors', 'Other']

        self.seg_class_description_dict = seg_class_description_dict
        self.seg_class_easy_name_dict = seg_class_easy_name_dict
        self.seg_class_shape_description_dict = seg_class_shape_description_dict
        self.seg_class_easy_description_dict = seg_class_easy_description_dict

    def preprocess_seg(self, seg_path):
        seg = cv2.imread(seg_path)
        if 'binary_masks' in seg_path:
            seg = seg // 255
        elif 'parts_masks' in seg_path:
            seg = seg // 85
        elif 'instruments_masks' in seg_path:
            seg = seg // 32
        return seg

    def run(self):
        image_path_list = get_file_list(self.image_dir)
        seg_path_list = get_file_list(self.seg_dir)
        image_path_list.sort()
        seg_path_list.sort()

        info_list = []

        for image_path, seg_path in zip(image_path_list, seg_path_list):
            print(image_path)
            image = cv2.imread(image_path)
            seg = self.preprocess_seg(seg_path)
            for seg_idx, seg_class in enumerate(self.seg_class_list):
                seg_idx_h, seg_idx_w = np.where(seg[:, :, 0] == seg_idx)
                if seg_idx_h.size > 0:
                    x1 = seg_idx_w.min()
                    y1 = seg_idx_h.min()
                    x2 = seg_idx_w.max()
                    y2 = seg_idx_h.max()
                    cropped_image = image[y1:y2, x1:x2]
                    mask = np.zeros(image.shape[:2])
                    mask[seg_idx_h, seg_idx_w] = 255
                    cropped_mask = mask[y1:y2, x1:x2]
                    mask_h, mask_w = np.where(cropped_mask != 255)
                    cropped_image_w_mask = copy.deepcopy(cropped_image)
                    cropped_image_w_mask[mask_h, mask_w] = 0
                    image_name = image_path.split('/')[-1]
                    image_name = image_name.replace('.jpg', '_{}.jpg'.format(seg_idx))
                    save_image_path = os.path.join(self.save_image_dir, image_name)
                    mask_name = image_name.replace('.jpg', '.png')
                    save_mask_path = os.path.join(self.save_mask_dir, mask_name)
                    save_image_w_mask_path = os.path.join(self.save_image_w_mask_dir, image_name)
                    cv2.imwrite(save_image_path, cropped_image)
                    cv2.imwrite(save_mask_path, cropped_mask)
                    cv2.imwrite(save_image_w_mask_path, cropped_image_w_mask)
                    info = {
                        'image_path': save_image_path,
                        'mask_path': save_mask_path,
                        'image_w_mask_path': save_image_w_mask_path,
                        'seg_class_name': seg_class,
                        'seg_class_description': self.seg_class_description_dict[seg_class],
                        'seg_class_easy_name': self.seg_class_easy_name_dict[seg_class],
                    }
                    info_list.append(info)

        if self.json_path is not None:
            fo = open(self.json_path, 'w')
            json.dump(info_list, fo)
        else:
            return info_list


if __name__ == '__main__':
    # process all data
    # if '/jmain02/home/J2AD019/exk01/zxz35-exk01' in os.getcwd():
    #     root_path = '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/cropped_train'  # jade2
    # else:
    #     root_path = '/scratch/grp/grv_shi/cambridge-1/data/EndoVis2017/cropped_train'  # create
    root_path='/content/drive/MyDrive/EndoVis2017-modified/cropped_train'
    seg_type_list = ['binary_masks', 'parts_masks', 'instruments_masks']
    json_path = os.path.join(root_path, 'train_instruments_clip.json')
    all_info_list = []
    for seg_type in seg_type_list[2:]:
        print('config: {}'.format(seg_type))
        for i in range(8):
            image_dir = os.path.join(root_path, 'instrument_dataset_{}'.format(i + 1), 'images')
            seg_dir = os.path.join(root_path, 'instrument_dataset_{}'.format(i + 1), seg_type)
            save_image_dir = os.path.join(root_path, 'instrument_dataset_{}'.format(i + 1), 'images_clip')
            save_mask_dir = os.path.join(root_path, 'instrument_dataset_{}'.format(i + 1), 'masks_clip')
            save_image_w_mask_dir = os.path.join(root_path, 'instrument_dataset_{}'.format(i + 1), 'images_w_masks_clip')
            client = PrepareData(image_dir, seg_dir, save_image_dir, save_mask_dir, save_image_w_mask_dir)
            info_list = client.run()
            all_info_list += info_list
    fo = open(json_path, 'w')
    json.dump(all_info_list, fo)
