import os
import random
import shutil
from multiprocessing import Pool

import cv2
import argparse
import numpy as np
from tqdm import trange

from api_usage.face_detect import FaceDet
from api_usage.face_alignment import FaceAlign
from api_usage.face_masker import FaceMasker

def separate_database(input_folder, num_cores):
    List_imgs = os.listdir(input_folder)
    Len_imgs = len(List_imgs)  # 数据集长度

    if num_cores == 1:  # 单核
        subset = List_imgs[:Len_imgs // 1]

        List_subsets = [subset]

    if num_cores == 2:  # 双核，将所有数据集分成两个子数据集
        subset1 = List_imgs[:Len_imgs // 2]
        subset2 = List_imgs[Len_imgs // 2:]

        List_subsets = [subset1, subset2]

    elif num_cores == 4:  # 四核，将所有数据集分成四个子数据集
        subset1 = List_imgs[:Len_imgs // 4]
        subset2 = List_imgs[Len_imgs // 4: Len_imgs // 2]
        subset3 = List_imgs[Len_imgs // 2: (Len_imgs * 3) // 4]
        subset4 = List_imgs[(Len_imgs * 3) // 4:]

        List_subsets = [subset1, subset2, subset3, subset4]

    elif num_cores >= 8:  # 八核以上，将所有数据集分成八个子数据集
        subset1 = List_imgs[:Len_imgs // 8]
        subset2 = List_imgs[Len_imgs // 8: Len_imgs // 4]
        subset3 = List_imgs[Len_imgs // 4: (Len_imgs * 3) // 8]
        subset4 = List_imgs[(Len_imgs * 3) // 8: Len_imgs // 2]
        subset5 = List_imgs[Len_imgs // 2: (Len_imgs * 5) // 8]
        subset6 = List_imgs[(Len_imgs * 5) // 8: (Len_imgs * 6) // 8]
        subset7 = List_imgs[(Len_imgs * 6) // 8: (Len_imgs * 7) // 8]
        subset8 = List_imgs[(Len_imgs * 7) // 8:]

        List_subsets = [subset1, subset2, subset3, subset4,
                        subset5, subset6, subset7, subset8]
    
    return List_subsets


def generate_mask(input_folder, output_folder, List_imgs, ratio, speed):
    # load model
    face_detector = FaceDet()
    face_align = FaceAlign()
    face_masker = FaceMasker(True)

    for i in trange(len(List_imgs)):
        subfolder = List_imgs[i]
        images_path = os.path.join(input_folder, subfolder)
        output_subfolder = os.path.join(output_folder, subfolder)

        if not os.path.exists(output_subfolder):
            os.mkdir(output_subfolder)

        images_path_list = os.listdir(images_path)  # os.listdir Having randomness
        pct = max(1, round(len(images_path_list) * ratio))  # Sampling by ratio, at least one, rounded off

        for file in images_path_list[:pct]:
            file_path = os.path.join(images_path, file)
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            bboxs = face_detector(image)
            face_lms = [np.reshape(face_align(image, box).astype(np.int32), (-1)) for box in bboxs]

            # face masker
            file_name, file_lx = file.split(".")
            save_dir = os.path.join(output_subfolder, file_name + "_masked." + file_lx)
            mask_template_name = str(random.randint(0, 7)) + ".png"
            
            # no face continue
            if len(face_lms) < 1:
                continue
            
            face_masker.add_mask_one(file_path, face_lms[0], mask_template_name, save_dir, speed) # masked one face

        for file in images_path_list[pct:]:
            file_path = os.path.join(images_path, file)
            save_dir = os.path.join(output_subfolder, file)
            shutil.copyfile(file_path, save_dir)  # Copy the image without adding a mask




def main():
    p = Pool(args.num_cores)
    List_subsets = separate_database(args.input_folder, args.num_cores)

    # 对每个进程分配工作
    for i in range(args.num_cores):
        p.apply_async(generate_mask, args=(args.input_folder, args.output_folder, List_subsets[i], args.random_ratio, args.speed))
    
    p.close()
    p.join()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input_folder', type=str, help='input path')
    p.add_argument('-r', '--random_ratio', type=float, default=1, help="ratio of masked face")
    p.add_argument('-o', '--output_folder', type=str, default="./output", help="output path")
    p.add_argument('-s', '--speed', type=int, default=1, help="1: Using Cython to speed up, 0: No speed up")
    p.add_argument('-c', '--num_cores', type=int, default=4, help='number of CPU cores')
    args = p.parse_args()

    if args.random_ratio > 1:
        args.random_ratio = 1
    elif args.random_ratio < 0:
        args.random_ratio = 0

    if not os.path.exists(args.input_folder):
        print("Input path not exist!")
        exit()

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    main()

