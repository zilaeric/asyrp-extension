import json
import os
import sys
from glob import glob
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# make imports work from the asyrp repository
sys.path.append("../lib/asyrp/")

from utils.text_dic import SRC_TRG_TXT_DIC
from IPython.utils import io

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

from torchmetrics.image.fid import FrechetInceptionDistance

device = "cuda:0"

RUNSPATH = "../runs/"


################
# REPRODUCTION #
################

def calculate_fid(image_folder_path_src, image_folder_path_target):

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    fid = FrechetInceptionDistance(feature=2048, normalize=True)

    images_src_path = glob(image_folder_path_src + "/*.png")
    images_target_path = glob(image_folder_path_target + "/*.png")

    images_src_tensors = []
    for image_path in tqdm(images_src_path, total=len(images_src_path), desc="parsing src images for fid"):
        src_img = Image.open(image_path)

        src_img_tensor = transform(src_img).to(torch.uint8)
        images_src_tensors.append(src_img_tensor)

    all_src_tensor = torch.stack(images_src_tensors, dim=0)
    fid.update(all_src_tensor, real=True)

    images_target_tensors = []
    for image_path in tqdm(images_target_path, total=len(images_target_path), desc="parsing target images for fid"):
        target_img = Image.open(image_path)

        target_img_tensor = transform(target_img).to(torch.uint8)
        images_target_tensors.append(target_img_tensor)
    
    all_target_tensor = torch.stack(images_target_tensors, dim=0)
    fid.update(all_target_tensor, real=False)

    fid_s = fid.compute()
    
    return fid_s.item()


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
        score_er = calculate_fid(path_edited, path_recon)
        score_ro = calculate_fid(path_original, path_recon)
        score_eo = calculate_fid(path_edited, path_original)
        score_oo = calculate_fid(path_original, path_original)
        

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


class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity


def reproduction_sdir(dt_lambda=1.0):
    attrs = [
        "smiling",
        "sad",
        "tanned",
        "pixar",
        "neanderthal"
    ]

    clip_id = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)

    results = {}
    for attr in attrs:
        path_original = f"{RUNSPATH}/{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/original/"
        path_edited = f"{RUNSPATH}/{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/"
        src_texts, target_texts = SRC_TRG_TXT_DIC[attr]

        list_src_path = glob(path_original + "/*.png")
        list_target_path = glob(path_edited + "/*.png")

        scores = []

        for src_path, target_path in zip(list_src_path, list_target_path):
            original_image = Image.open(src_path)
            edited_image = Image.open(target_path)

            similarity_score = dir_similarity(original_image, edited_image, src_texts, target_texts)
            scores.append(float(similarity_score.detach().cpu()))

        s_dir = np.mean(scores)
        print(f"Attribute {attr} gives CLIP directional similarity: {s_dir}")

        results[attr] = {
            "S_dir": s_dir
        }

    results_json = json.dumps(results, indent=4)

    # Writing to sample.json
    with open(f"{RUNSPATH}/reproduction_sdir.json", "w") as f:
        f.write(results_json)