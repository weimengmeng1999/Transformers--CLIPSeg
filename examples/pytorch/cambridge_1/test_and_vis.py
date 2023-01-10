import os
import sys
import cv2
import copy
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPVisionModel, CLIPProcessor, CLIPTokenizer
from examples.pytorch.cambridge_1.utils import (seg_class_description_dict,
                                                seg_class_easy_name_dict,
                                                seg_class_shape_description_dict,
                                                seg_class_easy_description_dict)
from examples.pytorch.cambridge_1.utils import get_file_list


class TestAndVis(object):
    def __init__(self,
                 image_dir=None,
                 seg_dir=None,
                 vis_dir=None,
                 text_type='seg_class_name',
                 attention_type='dot_product',
                 model_name_or_path="openai/clip-vit-base-patch32"):

        # config
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.vis_dir = vis_dir
        self.text_type = text_type
        self.attention_type = attention_type

        if seg_dir is None:
            seg_dir = 'binary_masks'  # in case there is a bug.
        # palette: BGR
        if 'binary_masks' in seg_dir:
            self.seg_class_list = ['Background', 'Instrument']
            self.seg_palette = [(255, 128, 0), (0, 0, 255)]
        elif 'parts_masks' in seg_dir:
            self.seg_class_list = ['Background', 'Shaft', 'Wrist', 'Claspers']
            self.seg_palette = [(255, 128, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
        elif 'instruments_masks' in seg_dir:
            self.seg_class_list = ['Background', 'Bipolar Forceps', 'Prograsp Forceps', 'Large Needle Driver',
                                   'Vessel Sealer', 'Grasping Retractor', 'Monopolar Curved Scissors', 'Other']
            self.seg_palette = [(255, 128, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                                (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255)]
        else:
            self.seg_class_list = ['Background', 'Instrument']
            self.seg_palette = [(255, 128, 0), (0, 0, 255)]

        self.seg_class_description_dict = seg_class_description_dict
        self.seg_class_easy_name_dict = seg_class_easy_name_dict
        self.seg_class_shape_description_dict = seg_class_shape_description_dict
        self.seg_class_easy_description_dict = seg_class_easy_description_dict

        self.model = CLIPModel.from_pretrained(model_name_or_path)
        self.image_model = CLIPVisionModel.from_pretrained(model_name_or_path)
        self.image_processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.text_tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)

    def forward_image_single(self, image_path):
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.image_model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output  # pooled CLS states
        return last_hidden_state, pooled_output

    def forward_text_single(self, text):
        inputs = self.text_tokenizer([text], padding=True, return_tensors="pt")
        new_inputs = {
            'input_ids': inputs['input_ids'][:, :self.text_tokenizer.model_max_length],
            'attention_mask': inputs['attention_mask'][:, :self.text_tokenizer.model_max_length],
        }
        text_features = self.model.get_text_features(**new_inputs)
        return text_features

    def preprocess_seg(self, seg_path):
        seg = cv2.imread(seg_path)
        if 'binary_masks' in seg_path:
            seg = seg // 255
        elif 'parts_masks' in seg_path:
            seg = seg // 85
        elif 'instruments_masks' in seg_path:
            seg = seg // 32
        return seg

    def ndarray_norm(self, ndarray):
        ndarray_min = ndarray.min()
        ndarray_max = ndarray.max()
        return (ndarray - ndarray_min) / (ndarray_max - ndarray_min)

    def get_attention_map(self, image_feature, text_feature):
        ''' Input shape:
        image_feature: (h, w, c)
        text_feature: (c)
        '''
        if self.attention_type == 'dot_product':
            # dot product
            attention_map = torch.einsum('hwc,c->hw', image_feature, text_feature).sigmoid().cpu().detach().numpy()
        elif self.attention_type in ['cos', 'l1', 'l2']:
            # distance/similarity
            h, w = image_feature.shape[:2]
            attention_map = torch.zeros((h, w))
            for y in range(h):
                for x in range(w):
                    if self.attention_type == 'cos':
                        value = F.cosine_similarity(image_feature[y, x], text_feature, dim=0)
                    elif self.attention_type == 'l1':
                        value = F.pairwise_distance(image_feature[y, x], text_feature, p=1.0)
                    elif self.attention_type == 'l2':
                        value = F.pairwise_distance(image_feature[y, x], text_feature, p=2.0)
                    attention_map[y, x] = value
            attention_map = attention_map.cpu().detach().numpy()
            attention_map = self.ndarray_norm(attention_map)
        elif self.attention_type in ['only_image_feature_0', 'only_image_feature_mean']:
            if self.attention_type == 'only_image_feature_0':
                attention_map = image_feature[:, :, 0].cpu().detach().numpy()
            elif self.attention_type == 'only_image_feature_mean':
                attention_map = torch.mean(image_feature, dim=2).cpu().detach().numpy()
            attention_map = self.ndarray_norm(attention_map)
        attention_map = 1. - attention_map
        # filter low value
        attention_map_flatten = attention_map.reshape(-1)
        thresh = np.sort(attention_map_flatten)[-int(attention_map_flatten.size * 0.25)]
        a_y, a_x = np.where(attention_map < thresh)
        attention_map[a_y, a_x] = 0
        return attention_map

    def process_one_image_text(self, image_path, seg_path, text, seg_idx):
        # 1. get image feature
        image_last_hidden_state, image_pooled_output = self.forward_image_single(image_path)
        image_last_hidden_state = self.model.visual_projection(image_last_hidden_state)
        attention_map_size = int(np.sqrt(image_last_hidden_state.shape[1] - 1))
        image_feature = image_last_hidden_state[0, 1:].reshape((attention_map_size, attention_map_size, -1))
        # 2. get text feature
        text_feature = self.forward_text_single(text)[0]
        # 3. get attention map
        attention_map = self.get_attention_map(image_feature, text_feature)
        # 4. vis
        image_cat = self.vis(image_path, seg_path, attention_map, seg_idx)
        # 5. save vis (write to current dir)
        save_name = image_path.split('/')[-1]
        save_name = save_name.replace('.jpg', '_{}_{}.jpg'.format(seg_idx, self.seg_class_list[seg_idx].replace(' ', '_')))
        cv2.imwrite(save_name, image_cat)

    def run_one_image(self, image_path, seg_path, show_vis=True, save_vis=False):
        # 1. get image feature
        image_last_hidden_state, image_pooled_output = self.forward_image_single(image_path)
        image_last_hidden_state = self.model.visual_projection(image_last_hidden_state)
        attention_map_size = int(np.sqrt(image_last_hidden_state.shape[1] - 1))
        image_feature = image_last_hidden_state[0, 1:].reshape((attention_map_size, attention_map_size, -1))
        # 2. iterate over different seg classes
        for seg_idx, seg_class in enumerate(self.seg_class_list):
            if self.text_type == 'seg_class_name':
                text = seg_class
            else:
                text = getattr(self, '{}_dict'.format(self.text_type))[seg_class]
            # 3. get text feature
            text_feature = self.forward_text_single(text)[0]
            # 4. get attention map
            attention_map = self.get_attention_map(image_feature, text_feature)
            # 5. vis
            image_cat = self.vis(image_path, seg_path, attention_map, seg_idx)
            # 6. show or save vis (write to vis_dir)
            if show_vis:
                self.show_vis(image_cat, image_path, seg_idx)
            if save_vis:
                self.save_vis(image_cat, image_path, seg_idx)

    def run(self):
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        image_path_list = get_file_list(self.image_dir)
        seg_path_list = get_file_list(self.seg_dir)
        image_path_list.sort()
        seg_path_list.sort()

        for idx, (image_path, seg_path) in enumerate(zip(image_path_list, seg_path_list)):
            if idx > 5:
                break
            print(image_path)
            self.run_one_image(image_path, seg_path, show_vis=False, save_vis=True)

    def vis(self, image_path, seg_path, attention_map, seg_idx):
        image = cv2.imread(image_path)
        seg = self.preprocess_seg(seg_path)
        h, w, c = image.shape
        # attention_map = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)
        attention_map = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_NEAREST)
        attention_map_new = np.zeros(image.shape)
        for i in range(3):
            attention_map_new[:, :, i] = attention_map
        seg_idx_h, seg_idx_w = np.where(seg[:, :, 0] == seg_idx)
        image_1 = copy.deepcopy(image)
        image_1[seg_idx_h, seg_idx_w] = image_1[seg_idx_h, seg_idx_w] * 0.5 + [x * 0.5 for x in self.seg_palette[seg_idx]]
        image_2 = copy.deepcopy(image)
        image_2 = image_2 * (1 - attention_map_new) + self.seg_palette[seg_idx] * attention_map_new
        image_cat = np.concatenate((image_1, image_2), axis=1)
        return image_cat

    def show_vis(self, image_cat, image_path, seg_idx):
        show_name = image_path.replace(self.image_dir, self.vis_dir)
        show_name = show_name.replace('.jpg', '_{}_{}.jpg'.format(seg_idx, self.seg_class_list[seg_idx].replace(' ', '_')))
        cv2.imshow(show_name, image_cat / 255.)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_vis(self, image_cat, image_path, seg_idx):
        save_path = image_path.replace(self.image_dir, self.vis_dir)
        save_path = save_path.replace('.jpg', '_{}_{}.jpg'.format(seg_idx, self.seg_class_list[seg_idx].replace(' ', '_')))
        cv2.imwrite(save_path, image_cat)


if __name__ == '__main__':
    if '/jmain02/home/J2AD019/exk01/zxz35-exk01' in os.getcwd():
        root_path = '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/cropped_train'  # jade2
    else:
        root_path = '/scratch/grp/grv_shi/cambridge-1/data/EndoVis2017/cropped_train'  # create

    process_all = True
    if not process_all:
        # test one image with one text
        # image_path = '{}/instrument_dataset_1/images_w_masks_clip/frame000_1.jpg'.format(root_path)
        # seg_path = '{}/instrument_dataset_1/images_w_masks_clip/frame000_1.jpg'.format(root_path)
        image_name = 'frame000_1_w_mask.jpg'
        image_path = '/users/k21163430/workspace/transformers/{}'.format(image_name)
        seg_path = '/users/k21163430/workspace/transformers/{}'.format(image_name)
        text = 'Bipolar Forceps'
        # model_name_or_path = 'openai/clip-vit-base-patch16'
        # model_name_or_path = '/users/k21163430/workspace/transformers/clip-vit-base-patch16-cambridge1-instruments-image_path-seg_class_name-finetune_fc'
        model_name_or_path = '/users/k21163430/workspace/transformers/clip-vit-base-patch16-cambridge1-instruments-image_w_mask_path-seg_class_name-finetune_fc'
        client = TestAndVis(text_type='seg_class_name', attention_type='dot_product', model_name_or_path=model_name_or_path)
        client.process_one_image_text(image_path, seg_path, text, 1)
        print(image_path)
    else:
        # process all data
        image_dir_name = 'images_clip'
        seg_type_list = ['images_clip']
        # seg_type_list = ['binary_masks', 'parts_masks', 'instruments_masks']
        text_type_list = ['seg_class_name', 'seg_class_description', 'seg_class_easy_name', 'seg_class_shape_description', 'seg_class_easy_description']
        attention_type_list = ['dot_product', 'cos', 'l1', 'l2', 'only_image_feature_0', 'only_image_feature_mean']
        model_name_or_path_list = ['openai/clip-vit-base-patch32', 'openai/clip-vit-base-patch16', 'openai/clip-vit-large-patch14', 'openai/clip-vit-large-patch14-336']
        # model_name_or_path_list = ['/users/k21163430/workspace/transformers/clip-vit-base-patch16-cambridge1-instruments-image_path-seg_class_description-finetune_fc']
        # model_name_or_path_list += ['/users/k21163430/workspace/transformers/clip-vit-base-patch16-cambridge1-instruments-image_path-seg_class_easy_name-finetune_fc']
        # model_name_or_path_list += ['/users/k21163430/workspace/transformers/clip-vit-base-patch16-cambridge1-instruments-image_path-seg_class_name-finetune_fc']
        # model_name_or_path_list += ['/users/k21163430/workspace/transformers/clip-vit-base-patch16-cambridge1-instruments-image_w_mask_path-seg_class_description-finetune_fc']
        # model_name_or_path_list += ['/users/k21163430/workspace/transformers/clip-vit-base-patch16-cambridge1-instruments-image_w_mask_path-seg_class_easy_name-finetune_fc']
        # model_name_or_path_list += ['/users/k21163430/workspace/transformers/clip-vit-base-patch16-cambridge1-instruments-image_w_mask_path-seg_class_name-finetune_fc']
        for seg_type in seg_type_list[0:1]:
            for text_type in text_type_list[0:3]:
                for attention_type in attention_type_list[0:1]:
                    for model_name_or_path in model_name_or_path_list[1:2]:
                        if 'openai/' in model_name_or_path:
                            model_tag = model_name_or_path.replace('openai/', '').replace('-', '_')
                        else:
                            model_tag = model_name_or_path.split('/')[-1].replace('-', '_')

                        print('config: {}, {}, {}, {}'.format(seg_type, text_type, attention_type, model_tag))
                        for i in range(1):
                            image_dir = os.path.join(root_path, 'instrument_dataset_{}'.format(i + 1), image_dir_name)
                            seg_dir = os.path.join(root_path, 'instrument_dataset_{}'.format(i + 1), seg_type)
                            vis_dir = os.path.join(root_path, 'instrument_dataset_{}'.format(i + 1), 'vis_{}_{}_{}_{}'.format(seg_type, text_type, attention_type, model_tag))
                            client = TestAndVis(image_dir, seg_dir, vis_dir, text_type, attention_type, model_name_or_path)
                            client.run()
