import sys
sys.path.append('.')

import torch
import yaml
import numpy as np
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

class FaceAlign():
    def __init__(self):
         # common setting for all model, need not modify.
        model_path = 'models'

        # model setting, modified along with model
        scene = 'non-mask'
        model_category = 'face_alignment'
        model_name =  model_conf[scene][model_category]

        print('Start to load the face landmark model...')
        # load model
        try:
            faceAlignModelLoader = FaceAlignModelLoader(model_path, "cuda" if torch.cuda.is_available() else "cpu", model_category, model_name)
        except Exception as e:
            print('Failed to parse model configuration file!')
            print(e)
            sys.exit(-1)
        else:
            print('Successfully parsed the model configuration file model_meta.json!')

        try:
            model, cfg = faceAlignModelLoader.load_model()
        except Exception as e:
            print('Model loading failed!')
            print(e)
            sys.exit(-1)
        else:
            print('Successfully loaded the face landmark model!\n')

        self.faceAlignModelHandler = FaceAlignModelHandler(model, "cuda" if torch.cuda.is_available() else "cpu", cfg)
 

    def __call__(self, image, bbox):
        det = np.asarray(list(map(int, bbox[0:4])), dtype=np.int32)
        landmarks = self.faceAlignModelHandler.inference_on_image(image, det) 
        return landmarks