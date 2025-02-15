from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import json
import tempfile
import pandas as pd
from os.path import join, splitext, exists
from collections import OrderedDict
from .mydataloader_retrieval import RetrievalDataset

class CharadesDataset(RetrievalDataset):
    """"Charades dataset."""
    def __init__(self,subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(CharadesDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                              max_frames, video_framerate, image_resolution, mode, config=config)
        pass

    def _get_anns(self, subset='train'):

        txt_path = {'train': join(self.anno_path, 'charades_sta_train.txt'),
                    'val': join(self.anno_path, 'charades_sta_test.txt'),
                    'test': join(self.anno_path, 'charades_sta_test.txt'),
                    'train_test': join(self.anno_path, 'charades_sta_train.txt')}[subset]
        
        if exists(txt_path):
            L = open(txt_path, 'r').readlines()
        else:
            raise FileNotFoundError
        
        L = [x.strip().split('##') for x in L]
        text = [x[1] for x in L]
        L = [x[0].split(' ') for x in L]
        ID = [x[0] for x in L]
        dur = [(float(x[1]), float(x[2])) for x in L]
        dur = [(x[1], x[0]) if x[0]>x[1] else (x[0], x[1]) for x in dur]

        video_path_dict = {}
        for root, dub_dir, video_files in os.walk(self.video_path):
            for video_file in video_files:
                video_id = video_file.split('.')[0]
                video_path_dict[video_id] = join(self.video_path, video_file)
        
        video_dict = OrderedDict()
        sentences_dict = OrderedDict()
        temp_dict = {}

        for i in range(len(ID)):
            if ID[i] in video_path_dict:
                video_dict[ID[i]] = video_path_dict[ID[i]]
                if ID[i] in temp_dict:
                    temp_dict[ID[i]].append((text[i], None, None))
                else:
                    temp_dict[ID[i]] = [(text[i], None, None)]
        for video_id, descriptions in temp_dict.items():
            sentences_dict[len(sentences_dict)] = (video_id, descriptions)
        return video_dict, sentences_dict