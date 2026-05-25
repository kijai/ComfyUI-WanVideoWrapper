# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from .arcface import Backbone
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    "FaceEncoderArcFace",
    "get_landmarks_from_image",
]


detector = None

def get_landmarks_from_image(image):
    """
    Detect landmarks with insightface.

    Args:
        image (np.ndarray or PIL.Image):
            The input image in RGB format.

    Returns:
        5 2D keypoints, only one face will be returned.
    """
    from insightface.app import FaceAnalysis
    global detector
    if detector is None:
        detector = FaceAnalysis()
        detector.prepare(ctx_id=0, det_size=(640, 640))

    in_image = np.array(image).copy()
    
    faces = detector.get(in_image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    
    # Get the largest face
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    
    # Return the 5 keypoints directly
    keypoints = face.kps  # 5 x 2

    return keypoints


def load_file_from_url(url, model_dir=None, progress=True, file_name=None, save_dir=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, model_dir)
    os.makedirs(save_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(save_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def init_recognition_model(model_name, half=False, device='cuda', model_rootpath=None):
    print("Initializing recognition model:", model_name)
    if model_name == 'arcface':
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').to('cuda').eval()
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/recognition_arcface_ir_se50.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # Always use ComfyUI models directory
    save_dir = os.path.join(model_rootpath, 'facexlib', 'weights')
    model_path = load_file_from_url(
        url=model_url, model_dir=None, progress=True, file_name=None, save_dir=save_dir)

    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model

class FaceEncoderArcFace():
    """ Official ArcFace, no_grad-only """

    def __repr__(self):
        return "ArcFace"

    def init_encoder_model(self, device, eval_mode=True, model_rootpath=None):
        self.device = device
        self.encoder_model = init_recognition_model('arcface', device=device, model_rootpath=model_rootpath)

        if eval_mode:
            self.encoder_model.eval()

    def __call__(self, in_image):
        return self.encoder_model(in_image[:, [2, 1, 0], :, :].contiguous()) # [B, 512], normalized