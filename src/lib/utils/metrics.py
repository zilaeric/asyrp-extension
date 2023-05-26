import json
import os
import sys
from glob import glob
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_fid import fid_score
from tqdm import tqdm

# make imports work from the asyrp repository
sys.path.append("../lib/asyrp/")

from losses.clip_loss import CLIPLoss
from utils.text_dic import SRC_TRG_TXT_DIC
from IPython.utils import io

device = "cuda:0"

clip_loss_func = CLIPLoss(
    device,
    lambda_direction=1,
    lambda_patch=0,
    lambda_global=0,
    lambda_manifold=0,
    lambda_texture=0,
    direction_loss_type='cosine',
    clip_model='ViT-B/32'
)

RUNSPATH = "../runs/"


################
# REPRODUCTION #
################

def reproduction_sdir(dt_lambda=1.0):
    attrs = [
        "smiling",
        "sad",
        "tanned",
        "pixar",
        "neanderthal"
    ]

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    for attr in attrs:
        src_texts, target_texts = SRC_TRG_TXT_DIC[attr]

        img_paths_original = glob(f"{RUNSPATH}/{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/original/*.png")
        img_paths_reconstructed = glob(f"{RUNSPATH}/{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/reconstructed/*.png")
        img_paths_edited = glob(f"{RUNSPATH}/{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/*.png")

        S_dir = 0
        dir_loss = 0

        results = {}
        with torch.no_grad():
            for i in range(len(img_paths_original)):
                src_img = Image.open(img_paths_reconstructed[i])
                target_img = Image.open(img_paths_edited[i])

                src_img_tensor = transform(src_img).float().to(device)[None, :]
                target_img_tensor = transform(target_img).float().to(device)[None, :]

                S_dir += clip_loss_func.clip_directional_loss(src_img_tensor, src_texts, target_img_tensor, target_texts)
                dir_loss += clip_loss_func.direction_loss(src_img_tensor, target_img_tensor).mean().cpu().detach().numpy()

                # clip_loss = -torch.log((2 - clip_loss_intermediate) / 2)
                # print(clip_loss)

                del src_img_tensor, target_img_tensor

        
        results[attr] = {
            "S_dir": S_dir / len(img_paths_edited),
            "dir_loss": 1-dir_loss / len(img_paths_edited)
        }

    results_json = json.dumps(results, indent=4)

    # Writing to sample.json
    with open(f"{RUNSPATH}/reproduction_sdir.json", "w") as f:
        f.write(results_json)


def reproduction_fid(dt_lambda=0.5):
    attrs = [
        "smiling",
        "sad",
        "tanned",
        "pixar",
        "neanderthal"
    ]

    results = {}
    for attr in attrs:
        path_original = f"{RUNSPATH}/{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/original/"
        path_recon = f"{RUNSPATH}/{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/reconstructed/"
        path_edited = f"{RUNSPATH}/{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/"

        # fid scores
        with io.capture_output() as captured:
            score_er = fid_score.calculate_fid_given_paths(
                [path_edited, path_recon],
                batch_size=32,
                device="cuda:0",
                dims=2048
            )
            score_eo = fid_score.calculate_fid_given_paths(
                [path_edited, path_original],
                batch_size=32,
                device="cuda:0",
                dims=2048
            )
            score_ro = fid_score.calculate_fid_given_paths(
                [path_recon, path_original],
                batch_size=32,
                device="cuda:0",
                dims=2048
            )
            score_oo = fid_score.calculate_fid_given_paths(
                [path_original, path_original],
                batch_size=32,
                device="cuda:0",
                dims=2048
            )

        print(f"Attribute {attr} gives FID: {score_er} between edited and reconstructed")
        print(f"Attribute {attr} gives FID: {score_eo} between edited and original")
        print(f"Attribute {attr} gives FID: {score_ro} between reconstructed and original")
        print(f"Attribute {attr} gives FID: {score_oo} between original and original")
        print("=" * 50)

        results[attr] = {
            "score_er": score_er,
            "score_eo": score_eo,
            "score_ro": score_ro,
            "score_oo": score_oo
        }

    results_json = json.dumps(results, indent=4)

    # Writing to sample.json
    with open(f"{RUNSPATH}/reproduction_sdir.json", "w") as f:
        f.write(results_json)
