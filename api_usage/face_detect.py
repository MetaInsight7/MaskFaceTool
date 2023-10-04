import sys
sys.path.append('.')

import torch
import yaml
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

class FaceDet():
    def __init__(self):
        # common setting for all model, need not modify.
        model_path = 'models'

        # model setting, modified along with model
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name =  model_conf[scene][model_category]

        print('Start to load the face detection model...')
        # load model
        try:
            faceDetModelLoader = FaceDetModelLoader(model_path, "cuda:0" if torch.cuda.is_available() else "cpu", model_category, model_name)
        except Exception as e:
            print('Failed to parse model configuration file!')
            print(e)
            sys.exit(-1)
        else:
            print('Successfully parsed the model configuration file model_meta.json!')

        try:
            model, cfg = faceDetModelLoader.load_model()
        except Exception as e:
            print('Model loading failed!')
            print(e)
            sys.exit(-1)
        else:
            print('Successfully loaded the face detection model!\n')

        self.faceDetModelHandler = FaceDetModelHandler(model, "cuda:0" if torch.cuda.is_available() else "cpu", cfg) 
    
    def __call__(self, image):
        try:
            dets = self.faceDetModelHandler.inference_on_image(image)
        except Exception as e:
            print('Face detection failed!')
            print(e)
            sys.exit(-1) 
        bboxs = dets 
        return bboxs