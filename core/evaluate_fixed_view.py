# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import numpy as np
import os, sys
import torch
import torch.backends.cudnn
import torch.utils.data
import cv2

from shutil import copyfile

import utils.point_cloud_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder

from pyntcloud import PyntCloud

def evaluate_fixed_view_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

    eval_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
    
    # Load weight
    # Load weight for encoder, decoder
    print('[INFO] %s Loading reconstruction weights from %s ...' % (dt.now(), cfg.EVALUATE_FIXED_VIEW.RECONSTRUCTION_WEIGHTS))
    rec_checkpoint = torch.load(cfg.EVALUATE_FIXED_VIEW.RECONSTRUCTION_WEIGHTS)
    encoder.load_state_dict(rec_checkpoint['encoder_state_dict'])
    decoder.load_state_dict(rec_checkpoint['decoder_state_dict'])
    print('[INFO] Best reconstruction result at epoch %d ...' % rec_checkpoint['epoch_idx'])
    
    # load samples
    samples = []
    with open(cfg.EVALUATE_FIXED_VIEW.SAMPLE_FILE) as s_f:
        samples = s_f.readlines()

    # evaluate single img on fixed views
    views = []
    with open(cfg.EVALUATE_FIXED_VIEW.VIEW_FILE) as v_f:
        views = v_f.readlines()
    
    for sample in samples:
        sample =  sample.replace('\n','')
        print(sample)

        # create sample dir
        evaluate_sample_dir =  os.path.join(cfg.EVALUATE_FIXED_VIEW.RESULT_DIR, sample)
        if not os.path.exists(evaluate_sample_dir):
            os.mkdir(evaluate_sample_dir)

        # create input folder for sample
        evaluate_sample_input_dir = os.path.join(evaluate_sample_dir, "input_img")
        if not os.path.exists(evaluate_sample_input_dir):
            os.mkdir(evaluate_sample_input_dir)
        
        # create output folder for sample
        evaluate_sample_output_dir = os.path.join(evaluate_sample_dir, "output")
        if not os.path.exists(evaluate_sample_output_dir):
            os.mkdir(evaluate_sample_output_dir)
        
        # create gt dir
        evaluate_sample_output_gt_dir = os.path.join(evaluate_sample_output_dir, "gt")
        if not os.path.exists(evaluate_sample_output_gt_dir):
            os.mkdir(evaluate_sample_output_gt_dir)

        # get gt pointcloud
        gt_point_cloud_file = cfg.DATASETS.SHAPENET.POINT_CLOUD_PATH % (cfg.EVALUATE_FIXED_VIEW.TAXONOMY_ID, sample)
        gt_point_cloud = get_point_cloud(gt_point_cloud_file)
        
        # generate gt img
        for view_id, view in enumerate(views):
            fixed_view = view.split()
            fixed_view = [round(float(item)) if float(item) < 359.5 else 0 for item in fixed_view]
            fixed_view = np.array(fixed_view)
                
            # Predict Pointcloud
            rendering_views = utils.point_cloud_visualization.get_point_cloud_image(gt_point_cloud, 
                                                                                    evaluate_sample_output_gt_dir,
                                                                                    view_id, "ground truth", fixed_view)

        # generate fixed view output
        for img_id in range(0, 24):
            # get input img path
            input_img_path = cfg.DATASETS.SHAPENET.RENDERING_PATH % (cfg.EVALUATE_FIXED_VIEW.TAXONOMY_ID , sample, img_id)
            
            # Save a copy of input image
            copyfile(input_img_path, os.path.join(evaluate_sample_input_dir, str(img_id) + '.png'))

            g_pc = evaluate_on_fixed_view_img(cfg,
                                              encoder, decoder,
                                              input_img_path,
                                              eval_transforms)

            evaluate_sample_output_img_dir = os.path.join(evaluate_sample_output_dir, str(img_id))
            if not os.path.exists(evaluate_sample_output_img_dir):
                os.mkdir(evaluate_sample_output_img_dir)
           
            for view_id, view in enumerate(views):
                fixed_view = view.split()
                fixed_view = [round(float(item)) if float(item) < 359.5 else 0 for item in fixed_view]
                fixed_view = np.array(fixed_view)
                
                # Predict Pointcloud
                rendering_views = utils.point_cloud_visualization.get_point_cloud_image(g_pc,
                                                                                        evaluate_sample_output_img_dir,
                                                                                        view_id, "reconstruction", fixed_view)

            
def get_point_cloud(point_cloud_file):
    # get data of point cloud
    _, suffix = os.path.splitext(point_cloud_file)
    if suffix == '.ply':
        point_cloud = PyntCloud.from_file(point_cloud_file)
        point_cloud = np.array(point_cloud.points)

    return point_cloud


def evaluate_on_fixed_view_img(cfg,
                               encoder, decoder,
                               input_img_path,
                               eval_transforms):
    # load img
    img_np = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    sample = np.array([img_np])
    rendering_images = eval_transforms(rendering_images=sample)
    
    # inference model
    with torch.no_grad():
        # Only one image per sample
        rendering_images = torch.squeeze(rendering_images, 1)
        
         # Get data from data loader
        rendering_images = utils.network_utils.var_or_cuda(rendering_images)
        
        vgg_features, image_code = encoder(rendering_images)
        image_code = [image_code]
        generated_point_clouds = decoder(image_code)
    
    return generated_point_clouds[0].detach().cpu().numpy()
