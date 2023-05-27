# make imports work from the asyrp repository
import sys
sys.path.append("../lib/asyrp/")

# to count the number of files in a folder
from glob import glob

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance

from PIL import Image
from tqdm import tqdm
import json

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

# dictionary of source and target texts for all attributes
from utils.text_dic import SRC_TRG_TXT_DIC

# constants
DEVICE = "cuda:0"
RUNSPATH = "../runs/"


################
# REPRODUCTION #
################

def calculate_fid(source_path: str, target_path: str) -> float:
    """Calculate Frechet Inception Distance (FID) over two sets of images.

    Args:
        source_path (str): Path to folder containing the first set of images.
        target_path (str): Path to folder containing the second set of images.

    Returns:
        float: FID metric over the given sets of images.
    """

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    fid = FrechetInceptionDistance(feature=2048, normalize=True)

    images_src_path = glob(source_path + "/*.png")
    images_target_path = glob(target_path + "/*.png")

    images_src_tensors = []
    for i in tqdm(range(len(images_src_path)), total=len(images_src_path), desc="parsing src images for fid"):
        if "original" in source_path:
            image_path = source_path + f"test_{i}_19_ngen40_original.png"
        elif "reconstructed" in source_path:
            image_path = source_path + f"test_{i}_19_ngen40_reconstructed.png"
        elif "edited" in source_path:
            image_path = source_path + f"test_{i}_19_ngen40_edited.png"
            
        try:
            src_img = Image.open(image_path)
        except:
            src_img = Image.open(image_path.replace("19_ngen", "0_ngen"))

        src_img_tensor = transform(src_img).to(torch.uint8)
        images_src_tensors.append(src_img_tensor)

    all_src_tensor = torch.stack(images_src_tensors, dim=0)
    fid.update(all_src_tensor, real=True)

    images_target_tensors = []
    for i in tqdm(range(len(images_src_path)), total=len(images_target_path), desc="parsing target images for fid"):
        if "original" in target_path:
            image_path = target_path + f"test_{i}_19_ngen40_original.png"
        elif "reconstructed" in target_path:
            image_path = target_path + f"test_{i}_19_ngen40_reconstructed.png"
        elif "edited" in target_path:
            image_path = target_path + f"test_{i}_19_ngen40_edited.png"

        try:
            src_img = Image.open(image_path)
        except:
            src_img = Image.open(image_path.replace("19_ngen", "0_ngen"))

        target_img = Image.open(image_path)

        target_img_tensor = transform(target_img).to(torch.uint8)
        images_target_tensors.append(target_img_tensor)
    
    all_target_tensor = torch.stack(images_target_tensors, dim=0)
    fid.update(all_target_tensor, real=False)

    fid_s = fid.compute()
    
    return fid_s.item()


def reproduction_fid(dt_lambda=0.5, run_path=RUNSPATH):
    attrs = [
        "smiling",
        "sad",
        "tanned",
        "pixar",
        "neanderthal"
    ]

    results = {}
    for attr in attrs:
        run_path = Path(run_path)
        path_original = run_path / f"{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/original/"
        path_recon = run_path / f"{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/reconstructed/"
        path_edited = run_path / f"{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/"

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
    with open(str(run_path / f"reproduction_fid_{dt_lambda}_correctorder_norm.json"), "w") as f:
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
        return {"pixel_values": image.to(DEVICE)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(DEVICE)}

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


def reproduction_sdir(dt_lambda=1.0, run_paths=RUNSPATH):
    attrs = [
        "smiling",
        "sad",
        "tanned",
        "pixar",
        "neanderthal"
    ]

    clip_id = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(DEVICE)
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(DEVICE)

    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)

    results = {}
    for attr in attrs:
        run_path = Path(run_path)
        path_original = run_path / f"{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/original/"
        path_reconstructed = run_path / f"{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/reconstructed/"
        path_edited = run_path / f"{attr}_{dt_lambda}_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/40/edited/"
        src_texts, target_texts = SRC_TRG_TXT_DIC[attr]

        list_original_path = glob(path_original + "/*.png")

        scores_or = []
        scores_oe = []
        scores_re = []

        for i in range(len(list_original_path)):
            original_path = path_original + f"test_{i}_0_ngen40_original.png"
            reconstructed_path = path_reconstructed + f"test_{i}_0_ngen40_reconstructed.png"
            edited_path = path_edited + f"test_{i}_0_ngen40_edited.png"
            
            original_image = Image.open(original_path)
            reconstructed_image = Image.open(reconstructed_path)
            edited_image = Image.open(edited_path)

            similarity_score_or = dir_similarity(original_image, reconstructed_image, src_texts, src_texts)
            similarity_score_oe = dir_similarity(original_image, edited_image, src_texts, target_texts)
            similarity_score_re = dir_similarity(reconstructed_image, edited_image, src_texts, target_texts)

            scores_or.append(float(similarity_score_or.detach().cpu()))
            scores_oe.append(float(similarity_score_oe.detach().cpu()))
            scores_re.append(float(similarity_score_re.detach().cpu()))

        sdir_or = np.mean(scores_or)
        sdir_or_var = np.std(scores_or)
        sdir_oe = np.mean(scores_oe)
        sdir_oe_var = np.std(scores_oe)
        sdir_re = np.mean(scores_re)
        sdir_re_var = np.std(scores_re)

        print(f"Attribute {attr} gives original-reconstructed CLIP directional similarity: {sdir_or}")
        print(f"Attribute {attr} gives original-edited CLIP directional similarity: {sdir_oe}")
        print(f"Attribute {attr} gives reconstructed-edited CLIP directional similarity: {sdir_re}")

        results[attr] = {
            "sdir_or": float(sdir_or),
            "sdir_or_var": float(sdir_or_var),
            "sdir_oe": float(sdir_oe),
            "sdir_oe_var": float(sdir_oe_var),
            "sdir_re": float(sdir_re),
            "sdir_re_var": float(sdir_re_var)
        }

    results_json = json.dumps(results, indent=4)

    # Writing to sample.json
    with open(str(run_path / "reproduction_sdir.json"), "w") as f:
        f.write(results_json)