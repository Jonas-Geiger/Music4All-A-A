# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) HuggingFace Inc. team.
# Copyright (c) Jonas Geiger/Johannes Kepler University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multimodal classification for MM-IMDB and Music4All datasets.

This file is based on the HuggingFace mm-imdb example (utils_mmimdb.py):
https://huggingface.co/spaces/xmadai/1bit_llama3_instruct_xmad_qa_batch/blob/main/examples/research_projects/mm-imdb/utils_mmimdb.py

Modified by Jonas Geiger, 2025
"""

import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError, ImageFile
from torch import nn
from torch.utils.data import Dataset
import warnings

from transformers import MMBTForClassification, ViltForImagesAndTextClassification, ViltProcessor, AutoModel, AutoConfig, AutoTokenizer
from sklearn.svm import LinearSVC
import logging

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier

logger = logging.getLogger(__name__)

POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model_type == "mmbt":
            model = torchvision.models.resnet152(pretrained=True)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.model = nn.Sequential(*modules)
            self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[args.num_image_embeds])
        elif args.model_type == "vilt":
            self.processor = ViltProcessor.from_pretrained(args.model_name_or_path)
        elif args.model_type == "singlebranch":
            self.model = torchvision.models.resnet152(pretrained=True)
            modules = list(self.model.children())[:-1]
            self.model = nn.Sequential(*modules)

    def forward(self, x):
        if self.args.model_type == "mmbt":
            out = self.pool(self.model(x))
            out = torch.flatten(out, start_dim=2)
            out = out.transpose(1, 2).contiguous()
            return out
        elif self.args.model_type == "vilt":
            return x
        elif self.args.model_type == "singlebranch":
            return self.model(x).squeeze(-1).squeeze(-1)


class JsonlDataset(Dataset):
    def __init__(self, data_path, text_ids, image_ids, tokenizer, transforms, labels, max_seq_length, model_type, split='train'):
        self.data_dir = data_path
        self.tokenizer = tokenizer
        self.labels = labels
        self.n_classes = len(labels)
        self.max_seq_length = max_seq_length
        self.transforms = transforms
        self.split = split
        self.model_type = model_type
        self.text_ids = set(text_ids)
        self.image_ids = set(image_ids)
        self.data = self.load_data(data_path)
        logging.info(f"Loaded {len(self.data)} valid items")
        if model_type == "vilt":

            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
            self.vilt_transforms = get_vilt_image_transforms()
        elif model_type == "singlebranch":
            self.singlebranch_transforms = get_image_transforms()

    def get_ids(self):
        # Returns the unique IDs for each item in the dataset.
        return [item['id'] for item in self.data]

    def load_data(self, data_path):
        data = []
        all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".json")]
        for json_file in all_files:
            try:
                with open(json_file, 'r') as jf:
                    item = json.load(jf)
                item_id = os.path.splitext(os.path.basename(json_file))[0]
                # Always filter using the provided splits
                text_used = item_id in self.text_ids
                image_used = item_id in self.image_ids
                # Skip items if neither modality is supposed to be used.
                if not text_used and not image_used:
                    continue
                item['text_used'] = text_used
                item['image_used'] = image_used
                item['id'] = item_id
                data.append(item)
            except Exception as e:
                logging.error(f"Error reading {json_file}: {e}")
        return data

    def __getitem__(self, index):
        item = self.data[index]
        # Extract text: iterate over "plot" list to find the first non-empty entry.
        if item['text_used']:
            text = ""
            if "plot" in item and isinstance(item["plot"], list):
                for p in item["plot"]:
                    if p and p.strip():
                        text = p.strip()
                        break
        else:
            text = ""
        # Load image.
        if not item['image_used']:
            image = self.get_blank_image()
        else:
            image_path = os.path.join(self.data_dir, f"{item['id']}.jpeg")
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logging.error(f"Error loading image {image_path}: {e}")
                image = self.get_blank_image()
        target = torch.zeros(self.n_classes)

        label_idxs = [self.labels.index(genre) for genre in item["genres"] if genre in self.labels]
        target[label_idxs] = 1
        if self.model_type == "mmbt":
            result = self.prepare_mmbt_item(text, image, target)
        elif self.model_type in ["vilt", "singlebranch"]:
            result = self.prepare_vilt_singlebranch_item(text, image, target)
        else:
            result = {"text": text, "image": image, "label": target}
        result["text_used"] = item['text_used']
        result["image_used"] = item['image_used']
        return result

    def __len__(self):
        return len(self.data)

    def get_blank_image(self):
        if self.model_type == "vilt":
            return Image.new('RGB', (384, 384), color='black')
        else:
            return Image.new('RGB', (224, 224), color='black')

    def prepare_mmbt_item(self, text, image, label):
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        sentence = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        image = self.transforms(image)
        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "attention_mask": attention_mask,
            "image": image,
            "label": label,
        }

    def prepare_vilt_singlebranch_item(self, text, image, label):
        if self.model_type == "vilt":
            image = self.vilt_transforms(image)
            encoding = self.processor(images=image, text=text, return_tensors="pt", padding="max_length",
                                      truncation=True, max_length=self.max_seq_length)
            input_ids = encoding.input_ids.squeeze(0)
            attention_mask = encoding.attention_mask.squeeze(0)
            pixel_values = encoding.pixel_values.squeeze(0)
        else:
            image = self.singlebranch_transforms(image)
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            pixel_values = image
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "label": label,
        }


def collate_fn(batch):
    text_used = [item["text_used"] for item in batch] if "text_used" in batch[0] else [True] * len(batch)
    image_used = [item["image_used"] for item in batch] if "image_used" in batch[0] else [True] * len(batch)

    if "pixel_values" in batch[0]:  # ViLT or SingleBranch, or Precomputed
        # Create attention masks and modality masks to use in models
        batch_size = len(batch)
        text_modality_available = torch.tensor(text_used, dtype=torch.bool)
        image_modality_available = torch.tensor(image_used, dtype=torch.bool)

        text_modality_mask_float = text_modality_available.float()
        image_modality_mask_float = image_modality_available.float()

        input_ids = torch.stack([item["input_ids"] for item in batch])
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] if item["attention_mask"] is not None
                                      else torch.ones_like(item["input_ids"]) for item in batch])

        for i in range(batch_size):
            if not text_modality_available[i]:
                input_ids[i].zero_()
                if attention_mask is not None:
                    attention_mask[i].zero_()

            if not image_modality_available[i]:
                pixel_values[i].zero_()

        # Keep return signature
        return input_ids, attention_mask, pixel_values, labels, text_modality_mask_float, image_modality_mask_float

    else:
        lens = [len(row["sentence"]) for row in batch]
        bsz, max_seq_len = len(batch), max(lens)

        mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
        text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

        for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
            text_tensor[i_batch, :length] = input_row["sentence"]
            mask_tensor[i_batch, :length] = 1

        img_tensor = torch.stack([row["image"] for row in batch])
        tgt_tensor = torch.stack([row["label"] for row in batch])
        img_start_token = torch.stack([row["image_start_token"] for row in batch])
        img_end_token = torch.stack([row["image_end_token"] for row in batch])

        # Create modality masks
        text_modality_mask = torch.tensor(text_used, dtype=torch.float)
        image_modality_mask = torch.tensor(image_used, dtype=torch.float)

        # For missing text: use padding tokens for a more realistic simulation
        for i, t_used in enumerate(text_used):
            if not t_used:
                text_tensor[i] = torch.zeros_like(text_tensor[i])  # Use padding token (0)
                mask_tensor[i] = torch.zeros_like(mask_tensor[i])  # Set attention mask to 0

        return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor, text_modality_mask, image_modality_mask


class Custom_MMBTForClassification(nn.Module):
    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.mmbt = MMBTForClassification(config=config, transformer=transformer, encoder=encoder)

    def forward(self, input_ids, input_modal, attention_mask, modal_start_tokens, modal_end_tokens, return_dict=False,
                text_used=None, image_used=None):
        if text_used is not None and not any(text_used):
            input_ids = None
            attention_mask = None
        if image_used is not None and not any(image_used):
            input_modal = None
            modal_start_tokens = None
            modal_end_tokens = None

        outputs = self.mmbt(
            input_ids=input_ids,
            input_modal=input_modal,
            attention_mask=attention_mask,
            modal_start_tokens=modal_start_tokens,
            modal_end_tokens=modal_end_tokens,
            return_dict=return_dict
        )
        return outputs


class Custom_ViLTForClassification(nn.Module):
    def __init__(self, args, config, num_labels):
        super().__init__()
        self.vilt = ViltForImagesAndTextClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
        self.vilt.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, pixel_values, attention_mask, text_used=None, image_used=None):
        if text_used is not None and not any(text_used):
            input_ids = None
            attention_mask = None
        if image_used is not None and not any(image_used):
            pixel_values = None

        outputs = self.vilt(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits
        return (logits,) + tuple(outputs.values())[1:]


def get_mmimdb_labels():
    return ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror',
            'Music', 'Musical', 'Mystery', 'Reality-TV', 'Romance', 'Sci-Fi',
            'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']


def get_image_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.46777044, 0.44531429, 0.40661017],
            std=[0.12221994, 0.12145835, 0.14380469],
        ),
    ])


def get_vilt_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize((384, 384)),
        ]
    )


def vilt_normalize(image):
    return transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )(image)


class Music4AllDataset(Dataset):
    def __init__(self, data_dir, text_mbids, image_mbids, tokenizer, transforms, max_seq_length, model_type, data_type, split='train'):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transforms = transforms
        self.split = split
        self.model_type = model_type
        self.data_type = data_type
        self.text_mbids = set(text_mbids)
        self.image_mbids = set(image_mbids)

        logging.info(f"Initializing Music4AllDataset with {len(text_mbids)} text MBIDs and {len(image_mbids)} image MBIDs")

        self.genres = self.get_all_genres()
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}

        self.num_labels = len(self.genres)

        self.data = self.load_data()

        logging.info(f"Loaded {len(self.data)} valid albums/artists")
        logging.info(f"Found {self.num_labels} unique genres")

        if model_type == "vilt":
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
            self.image_transforms = get_vilt_image_transforms()
        elif model_type == "mmbt" or model_type == "singlebranch" or model_type == "svm":
            self.image_transforms = get_image_transforms()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_all_genres(self):
        all_genres = set()
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_dir, filename), 'r') as f:
                    data = json.load(f)
                    if 'artist_info' in data:
                        all_genres.update(data['artist_info']['artist']['genres'])
                    elif 'album_info' in data:
                        all_genres.update(data['album_info']['album']['genres'])
        return sorted(list(all_genres))

    def load_data(self):
        data = []
        total_files = 0
        used_files = 0
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                total_files += 1
                with open(os.path.join(self.data_dir, filename), 'r') as f:
                    item = json.load(f)
                    text_used = item['mbid'] in self.text_mbids
                    image_used = item['mbid'] in self.image_mbids
                    if not text_used and not image_used:
                        continue
                    used_files += 1
                    item['text_used'] = text_used
                    item['image_used'] = image_used
                    data.append(item)
        logging.info(f"Loaded {used_files}/{total_files} items for split '{self.split}'")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        mbid = item['mbid']

        if 'artist_info' in item:
            text = item['artist_info']['artist']['wiki']['summary'] if item['text_used'] else ""
            genres = item['artist_info']['artist']['genres']
        elif 'album_info' in item:
            text = item['album_info']['album']['wiki']['summary'] if item['text_used'] else ""
            genres = item['album_info']['album']['genres']

        label = torch.zeros(self.num_labels)
        for genre in genres:
            if genre in self.genre_to_idx:
                label[self.genre_to_idx[genre]] = 1

        image = self.load_image(mbid) if item['image_used'] else self.transforms(self.get_blank_image())

        if self.model_type == "mmbt":
            result = self.prepare_mmbt_item(text, image, label)
        else:
            result = self.prepare_vilt_singlebranch_item(text, image, label)

        result["text_used"] = item['text_used']
        result["image_used"] = item['image_used']
        return result

    def prepare_mmbt_item(self, text, image, label):
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            "image_start_token": input_ids[0],
            "image_end_token": input_ids[-1],
            "sentence": input_ids[1:-1],
            "attention_mask": attention_mask,
            "image": image,
            "label": label,
        }

    def prepare_vilt_singlebranch_item(self, text, image, label):
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": image,
            "label": label,
        }

    def load_image(self, mbid):
        """Loads the image file with the highest index for the given mbid."""
        image_files = [
            f for f in os.listdir(self.data_dir)
            if f.startswith(str(mbid)) and f.lower().endswith('.jpg')
        ]

        if len(image_files) == 1:
            selected_filename = image_files[0]
        else:
            # Find the file with the highest index _N.jpg
            highest_index = -1
            best_filename = None
            for fname in image_files:
                match = re.search(r'_(\d+)\.jpg$', fname, re.IGNORECASE)
                if match:
                    current_index = int(match.group(1))
                    if current_index >= highest_index:
                        highest_index = current_index
                        best_filename = fname

            selected_filename = best_filename

        image_path = os.path.join(self.data_dir, selected_filename)
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transforms(image)
        except (OSError, UnidentifiedImageError) as e:
            print(f"Warning: Corrupted image file encountered: {image_path}. Error: {e}")
            return self.transforms(self.get_blank_image())  # Return transformed blank image

    def get_blank_image(self):
        if self.model_type == "vilt":
            return Image.new('RGB', (384, 384), color='black')
        else:
            return Image.new('RGB', (224, 224), color='black')

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            if 'artist_info' in row:
                label_freqs.update(row['artist_info']['artist']['genres'])
            elif 'album_info' in row:
                label_freqs.update(row['album_info']['album']['genres'])
        return label_freqs


class LLaVaMMIMDBDataset(Dataset):
    def __init__(self, dataset_folder, data_dir_image, data_dir_text, text_ids, image_ids, labels, split='train'):
        self.labels = labels
        self.n_classes = len(labels)
        self.split = split

        suffix = "test" if split == 'eval' else "train"

        # Load CSV files for pre-extracted LLAVA features.
        text_csv_path = os.path.join(data_dir_text, f"llava_plot_first_latent_{suffix}.csv")
        image_csv_path = os.path.join(data_dir_image, f"llava_images_latent_{suffix}.csv")
        text_csv = pd.read_csv(text_csv_path, dtype={'item_id': str})
        image_csv = pd.read_csv(image_csv_path, dtype={'item_id': str})

        # Load NPZ files to get expected dimensions.
        text_npz = np.load(os.path.join(data_dir_text, "llava_plot_first_latent_tensors.npz"), allow_pickle=True)
        image_npz = np.load(os.path.join(data_dir_image, "llava_latent_tensors.npz"), allow_pickle=True)
        expected_text_dim = text_npz['values'][0].shape[1]
        expected_image_dim = image_npz['values'][0].shape[1]

        # Verify CSV columns.
        assert 'item_id' in text_csv.columns, "Text CSV missing item_id column"
        assert 'item_id' in image_csv.columns, "Image CSV missing item_id column"

        text_csv_ids = text_csv['item_id'].astype(str)
        image_csv_ids = image_csv['item_id'].astype(str)

        text_features = text_csv.drop('item_id', axis=1)
        image_features = image_csv.drop('item_id', axis=1)

        logger.info(f"Original dimensions - Text: {text_features.shape[1]}, Image: {image_features.shape[1]}")
        logger.info(f"Expected dimensions - Text: {expected_text_dim}, Image: {expected_image_dim}")

        # Trim columns if necessary.
        if text_features.shape[1] > expected_text_dim:
            logger.warning(f"Trimming text features from {text_features.shape[1]} to {expected_text_dim}")
            text_features = text_features.iloc[:, :expected_text_dim]
        if image_features.shape[1] > expected_image_dim:
            logger.warning(f"Trimming image features from {image_features.shape[1]} to {expected_image_dim}")
            image_features = image_features.iloc[:, :expected_image_dim]

        # Load MMIMDb data by iterating over individual JSON files in the dataset folder.
        mmimdb_data = []
        json_files = [f for f in os.listdir(dataset_folder) if f.endswith(".json")]
        for file in json_files:
            file_path = os.path.join(dataset_folder, file)
            try:
                with open(file_path, 'r') as f:
                    item = json.load(f)
                if 'id' not in item:
                    item['id'] = os.path.splitext(file)[0]
                mmimdb_data.append(item)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        id_to_genres = {str(item['id']): item['genres'] for item in mmimdb_data}

        # Determine valid IDs as the intersection of modality splits, CSV IDs, and JSON file IDs.
        valid_ids = set(text_ids)
        valid_ids = valid_ids & set(id_to_genres.keys())
        valid_ids = valid_ids & set(text_csv_ids) & set(image_csv_ids)

        self.valid_ids = np.array(list(valid_ids))

        text_mask = text_csv_ids.isin(self.valid_ids)
        image_mask = image_csv_ids.isin(self.valid_ids)
        self.text_features = text_features.loc[text_mask].values
        self.image_features = image_features.loc[image_mask].values

        # Get genres for each valid id.
        self.item_genres = [id_to_genres[id_] for id_ in self.valid_ids]

        n_expected = len(text_ids)
        n_actual = len(self.valid_ids)
        logger.info(f"Split: {split}")
        logger.info(f"Expected samples: {n_expected}")
        logger.info(f"Actual samples after filtering: {n_actual}")
        logger.info(f"Final feature dimensions - Text: {self.text_features.shape[1]}, Image: {self.image_features.shape[1]}")

        # Verify dimensions.
        assert self.text_features.shape[1] == expected_text_dim, "Text features dimension mismatch after filtering"
        assert self.image_features.shape[1] == expected_image_dim, "Image features dimension mismatch after filtering"

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        text_feat = torch.tensor(self.text_features[idx], dtype=torch.float32)
        image_feat = torch.tensor(self.image_features[idx], dtype=torch.float32)
        target = torch.zeros(self.n_classes)
        # Build multi-label target using genres.
        label_idxs = [self.labels.index(genre) for genre in self.item_genres[idx] if genre in self.labels]
        target[label_idxs] = 1
        return {
            "input_ids": text_feat,
            "attention_mask": None,
            "pixel_values": image_feat,
            "label": target
        }


class SingleBranchModel(nn.Module):
    def __init__(self, args, num_labels):
        """
        SingleBranchModel that can optionally use pre-extracted embeddings.
        Correctly determines embedding dimensions based on args flags.
        """
        super().__init__()
        self.use_llava = getattr(args, 'use_llava_features', False)
        self.use_blip = getattr(args, 'use_blip_features', False)
        self.use_clip = getattr(args, 'use_clip_features', False)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) if not (self.use_llava or self.use_blip or self.use_clip) else None

        self.text_embed_dim = None
        self.image_embed_dim = None
        feature_source = "Online Encoding (BERT/ResNet)"  # Default

        if self.use_llava:
            feature_source = "LLaVa"
            try:
                # Determine LLaVa dims
                text_npz_path = os.path.join(args.llava_features_dir_text, "llava_plot_first_latent_tensors.npz")
                image_npz_path = os.path.join(args.llava_features_dir_image, "llava_latent_tensors.npz")
                text_data = np.load(text_npz_path, allow_pickle=True)
                image_data = np.load(image_npz_path, allow_pickle=True)
                sample_text = text_data['values'][0]
                sample_image = image_data['values'][0]
                self.text_embed_dim = sample_text.shape[-1]  # Get last dimension
                self.image_embed_dim = sample_image.shape[-1]  # Get last dimension
            except Exception as e:
                logger.error(f"Error loading LLaVA feature dimensions from NPZ: {e}. Using fallback 768.")
                self.text_embed_dim = 768
                self.image_embed_dim = 768

        elif self.use_blip:
            feature_source = "BLIP"
            try:
                # Determine BLIP dims from CSV
                text_csv_path = os.path.join(args.blip_features_dir_text, f"blip_txt_latent_{args.split}.csv")  # Use split
                image_csv_path = os.path.join(args.blip_features_dir_image, f"blip_images_latent_{args.split}.csv")  # Use split
                text_df = pd.read_csv(text_csv_path, nrows=1, dtype={'item_id': str})
                image_df = pd.read_csv(image_csv_path, nrows=1, dtype={'item_id': str})
                self.text_embed_dim = text_df.drop(columns=['item_id'], errors='ignore').shape[1]
                self.image_embed_dim = image_df.drop(columns=['item_id'], errors='ignore').shape[1]
            except Exception as e:
                logger.error(f"Error loading BLIP feature dimensions from CSV: {e}. Using fallback 768.")
                self.text_embed_dim = 768
                self.image_embed_dim = 768


        elif self.use_clip:
            feature_source = "CLIP"
            try:
                # Determine CLIP dims from CSV
                base_clip_path = args.clip_features_dir_text
                data_type_prefix = f"clip_{args.music4all_data_type}" if args.dataset == 'music4all' else "clip"

                text_csv_path = os.path.join(base_clip_path, f"{data_type_prefix}_encoded_texts",
                                             f"{data_type_prefix}_txt_latent_train.csv")
                image_csv_path = os.path.join(base_clip_path, f"{data_type_prefix}_encoded_images",
                                              f"{data_type_prefix}_images_latent_train.csv")
                text_df = pd.read_csv(text_csv_path, nrows=1, dtype={'item_id': str})
                image_df = pd.read_csv(image_csv_path, nrows=1, dtype={'item_id': str})
                self.text_embed_dim = text_df.drop(columns=['item_id'], errors='ignore').shape[1]
                self.image_embed_dim = image_df.drop(columns=['item_id'], errors='ignore').shape[1]
            except Exception as e:
                logger.error(f"Error loading CLIP feature dimensions from CSV: {e}. Using fallback 768.")
                self.text_embed_dim = 768
                self.image_embed_dim = 768

        # Only initialize online encoders if NO precomputed features are used
        if not (self.use_llava or self.use_blip or self.use_clip):

            feature_source = "Online Encoding (BERT/ResNet)"
            try:
                # Load text encoder and determine dimension
                self.text_encoder = AutoModel.from_pretrained(args.model_name_or_path)
                self.text_embed_dim = self.text_encoder.config.hidden_size
            except Exception as e:
                logger.error(f"Failed to load text encoder {args.model_name_or_path}: {e}. Setting text dim to fallback 768.")
                self.text_encoder = None
                self.text_embed_dim = 768

            try:
                # Load image encoder and determine dimension
                image_encoder_full = torchvision.models.resnet152(pretrained=True)
                # Get feature dimension before removing final layer
                self.image_embed_dim = image_encoder_full.fc.in_features
                # Create encoder without final FC layer
                self.image_encoder = nn.Sequential(*list(image_encoder_full.children())[:-1])
                del image_encoder_full  # Free memory
            except Exception as e:
                logger.error(f"Failed to load ResNet image encoder: {e}. Setting image dim to fallback 2048.")
                self.image_encoder = None
                self.image_embed_dim = 2048  # Fallback ResNet dim
        else:
            # Ensure encoder attributes don't exist if using precomputed
            self.text_encoder = None
            self.image_encoder = None

        if self.text_embed_dim is None or self.image_embed_dim is None:
            raise ValueError("Could not determine text or image embedding dimensions. Check paths and feature availability.")

        logger.info(f"Using feature source: {feature_source}")
        logger.info(f"Determined text embedding dim: {self.text_embed_dim}")
        logger.info(f"Determined image embedding dim: {self.image_embed_dim}")

        # Use text embedding size as the target common dimension
        self.common_dim = self.text_embed_dim
        logger.info(f"Common projection dimension: {self.common_dim}")

        # Create projection layers with the CORRECT determined dimensions
        self.text_projection = nn.Linear(self.text_embed_dim, self.common_dim)
        self.image_projection = nn.Linear(self.image_embed_dim, self.common_dim)

        # Add normalization layers
        self.text_norm = nn.LayerNorm(self.common_dim)
        self.image_norm = nn.LayerNorm(self.common_dim)

        # Modality-invariant MLP
        mlp_hidden_dim = max(2048, self.common_dim * 2)
        logger.info(f"MLP hidden dimension: {mlp_hidden_dim}")
        self.modality_invariant_network = nn.Sequential(
            nn.Linear(self.common_dim, mlp_hidden_dim),  # Input is common_dim
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim // 2, self.common_dim),  # Output is common_dim
            nn.LayerNorm(self.common_dim),
        )

        # The final classification layer
        self.classifier = nn.Linear(self.common_dim, num_labels)

        logger.info(f"Initialized SingleBranchModel:")
        logger.info(f"  use_llava={self.use_llava}, use_blip={self.use_blip}")
        logger.info(f"  Text embedding dim: {self.text_embed_dim}")
        logger.info(f"  Image embedding dim: {self.image_embed_dim}")
        logger.info(f"  Common hidden size: {self.common_dim}")

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, text_used=None, image_used=None):
        """
        Forward pass for the SingleBranchModel:
          - If using BLIP, LLaVa, or CLIP features, `input_ids` and `pixel_values` are
            pre-extracted embeddings. We simply project them.
          - Otherwise, we run BERT and ResNet to get embeddings, then project them.
          - If one modality is missing, we use the available modality.
          - If both are present, we average embeddings.
        """
        # If both modalities are missing entirely, raise an error
        if text_used is not None and not any(text_used) and image_used is not None and not any(image_used):
            # Check if text_used or image_used are tensors, convert if necessary
            check_text_used = text_used.cpu().numpy() if isinstance(text_used, torch.Tensor) else text_used
            check_image_used = image_used.cpu().numpy() if isinstance(image_used, torch.Tensor) else image_used
            if not np.any(check_text_used) and not np.any(check_image_used):
                raise RuntimeError("Both text and image modalities are missing for all items in the batch. Cannot proceed.")

        # Identify batch size and device
        if input_ids is not None:
            batch_size = input_ids.size(0)
            device = input_ids.device
        elif pixel_values is not None:
            batch_size = pixel_values.size(0)
            device = pixel_values.device
        else:
            # This case should ideally not happen if the check above works
            raise RuntimeError("Cannot determine batch size: both input_ids and pixel_values are None.")

        text_embeddings = None  # Initialize
        image_embeddings = None  # Initialize

        # START MODIFICATION
        # Process based on whether precomputed features are used
        if self.use_llava or self.use_blip or self.use_clip:
            # Pre-extracted embeddings path
            if input_ids is not None:
                # "input_ids" already contain N-dim features
                text_emb_proj = self.text_projection(input_ids)
                text_embeddings = self.text_norm(text_emb_proj)

            if pixel_values is not None:
                # "pixel_values" already contain N-dim features
                img_emb_proj = self.image_projection(pixel_values)
                image_embeddings = self.image_norm(img_emb_proj)

        else:
            # Standard text + image encoders path
            # BERT for text
            if input_ids is not None and self.tokenizer is not None:  # Check if tokenizer available
                # Ensure input_ids are integer token IDs for BERT
                if input_ids.dtype != torch.long:
                    logger.warning(f"Input IDs dtype is {input_ids.dtype}, expected torch.long for BERT encoder. Trying to cast.")
                    input_ids = input_ids.long()

                text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls_text_emb = text_outputs.last_hidden_state[:, 0, :]
                text_emb_proj = self.text_projection(cls_text_emb)
                text_embeddings = self.text_norm(text_emb_proj)
            else:
                # text_embeddings remains None if input_ids not provided or no tokenizer
                pass

            # ResNet for images
            if pixel_values is not None:
                #  Ensure pixel_values are appropriate image tensors (e.g., [B, C, H, W])
                if len(pixel_values.shape) != 4:
                    logger.warning(f"Pixel values shape is {pixel_values.shape}, expected 4D tensor for image encoder.")
                try:
                    # shape: (batch_size, 2048, 1, 1) after resnet's avgpool
                    img_emb = self.image_encoder(pixel_values).squeeze(-1).squeeze(-1)
                    img_emb_proj = self.image_projection(img_emb)
                    image_embeddings = self.image_norm(img_emb_proj)
                except Exception as e:
                    logger.error(f"Error during online image encoding: {e}", exc_info=True)
            else:
                pass

        # Create final embeddings per sample, combining available modalities
        combined = torch.zeros(batch_size, self.common_dim, device=device)

        # Ensure modality masks are boolean numpy arrays for iteration
        np_text_used = text_used.cpu().numpy() if isinstance(text_used, torch.Tensor) else np.array(text_used)
        np_image_used = image_used.cpu().numpy() if isinstance(image_used, torch.Tensor) else np.array(image_used)

        for i in range(batch_size):
            # Determine presence based on the flags passed from collate_fn
            has_text = np_text_used[i] if np_text_used is not None else (text_embeddings is not None)
            has_image = np_image_used[i] if np_image_used is not None else (image_embeddings is not None)

            # Get the actual embeddings for this sample
            current_text_emb = text_embeddings[i] if text_embeddings is not None else None
            current_image_emb = image_embeddings[i] if image_embeddings is not None else None

            if has_text and has_image:
                if current_text_emb is not None and current_image_emb is not None:
                    combined[i] = (current_text_emb + current_image_emb) / 2
                elif current_text_emb is not None:
                    logger.warning(f"Sample {i}: Image modality expected but embedding missing. Using text only.")
                    combined[i] = current_text_emb
                elif current_image_emb is not None:
                    logger.warning(f"Sample {i}: Text modality expected but embedding missing. Using image only.")
                    combined[i] = current_image_emb
                else:
                    logger.error(f"Sample {i}: Both modalities expected but embeddings missing. Using zeros.")
                    combined[i] = torch.zeros(self.common_dim, device=device)  # Fallback to zeros

            elif has_text:
                if current_text_emb is not None:
                    combined[i] = current_text_emb
                else:
                    logger.error(f"Sample {i}: Text modality expected but embedding missing. Using zeros.")
                    combined[i] = torch.zeros(self.common_dim, device=device)  # Fallback

            elif has_image:
                if current_image_emb is not None:
                    combined[i] = current_image_emb
                else:
                    logger.error(f"Sample {i}: Image modality expected but embedding missing. Using zeros.")
                    combined[i] = torch.zeros(self.common_dim, device=device)  # Fallback
            else:
                logger.error(f"Sample {i} has no available modality flag set. Using zeros.")
                combined[i] = torch.zeros(self.common_dim, device=device)

        # Pass combined embeddings through the MLP
        x = self.modality_invariant_network(combined)
        logits = self.classifier(x)
        return (logits,)


class BlipMMIMDBDataset(Dataset):
    def __init__(self, dataset_folder, data_dir_image, data_dir_text, text_ids, image_ids, labels, split='train'):
        self.labels = labels
        self.n_classes = len(labels)
        self.split = split

        suffix = "test" if split == 'eval' else "train"

        # Load BLIP CSV files
        text_csv_path = os.path.join(data_dir_text, f"blip_txt_latent_{suffix}.csv")
        image_csv_path = os.path.join(data_dir_image, f"blip_images_latent_{suffix}.csv")
        text_csv = pd.read_csv(text_csv_path, dtype={'item_id': str})
        image_csv = pd.read_csv(image_csv_path, dtype={'item_id': str})

        print(f"Loaded text CSV: {text_csv_path} with shape {text_csv.shape}")
        print(f"Loaded image CSV: {image_csv_path} with shape {image_csv.shape}")

        text_csv_ids = text_csv['item_id']
        image_csv_ids = image_csv['item_id']
        text_features = text_csv.drop(columns=['item_id'])
        image_features = image_csv.drop(columns=['item_id'])

        # Load all JSON files from dataset_folder to build a mapping id -> genres.
        mmimdb_data = []
        json_files = [f for f in os.listdir(dataset_folder) if f.endswith(".json")]
        for file in json_files:
            file_path = os.path.join(dataset_folder, file)
            try:
                with open(file_path, 'r') as jf:
                    item = json.load(jf)
                if 'id' not in item:
                    item['id'] = os.path.splitext(file)[0]
                mmimdb_data.append(item)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        id_to_genres = {str(item['id']): item['genres'] for item in mmimdb_data}

        # Convert split IDs and CSV IDs to sets.
        ti = set(text_ids)
        ii = set(image_ids)
        tc = set(text_csv_ids)
        ic = set(image_csv_ids)
        jg = set(id_to_genres.keys())

        # Final intersection: IDs common to the split, both CSVs, and JSON files.
        final_ids = ti & ii & tc & ic & jg
        print(f"Final intersection (number of valid samples): {len(final_ids)}")
        print(f"Number of valid images (from CSVs): {len(final_ids)}")

        self.valid_ids = np.array(list(final_ids))
        text_mask = text_csv_ids.isin(self.valid_ids)
        image_mask = image_csv_ids.isin(self.valid_ids)
        self.text_features = text_features.loc[text_mask].values
        self.image_features = image_features.loc[image_mask].values
        self.item_genres = [id_to_genres[idx] for idx in self.valid_ids]

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        text_feat = torch.tensor(self.text_features[idx], dtype=torch.float32)
        image_feat = torch.tensor(self.image_features[idx], dtype=torch.float32)
        target = torch.zeros(self.n_classes, dtype=torch.float32)
        label_idxs = [self.labels.index(genre) for genre in self.item_genres[idx] if genre in self.labels]
        target[label_idxs] = 1.0
        return {
            "input_ids": text_feat,
            "attention_mask": None,
            "pixel_values": image_feat,
            "label": target
        }


class ClipMMIMDBDataset(Dataset):
    def __init__(self, dataset_folder, data_dir_image, data_dir_text, text_ids, image_ids, labels, split='train'):
        self.labels = labels
        self.n_classes = len(labels)
        self.split = split
        self.text_ids_set = set(text_ids)
        self.image_ids_set = set(image_ids)

        suffix = "test" if split == 'eval' else "train"

        # Load CLIP CSV files
        text_csv_path = os.path.join(data_dir_text, f"clip_txt_latent_{suffix}.csv")
        image_csv_path = os.path.join(data_dir_image, f"clip_images_latent_{suffix}.csv")
        text_csv = pd.read_csv(text_csv_path, dtype={'item_id': str})
        image_csv = pd.read_csv(image_csv_path, dtype={'item_id': str})

        print(f"Loaded text CSV: {text_csv_path} with shape {text_csv.shape}")
        print(f"Loaded image CSV: {image_csv_path} with shape {image_csv.shape}")

        text_features_dict = text_csv.set_index('item_id').iloc[:, :].to_dict('index')
        image_features_dict = image_csv.set_index('item_id').iloc[:, :].to_dict('index')
        text_csv_ids = set(text_features_dict.keys())
        image_csv_ids = set(image_features_dict.keys())

        # Load all JSON files from dataset_folder to build a mapping id -> genres
        mmimdb_data = []
        json_files = [f for f in os.listdir(dataset_folder) if f.endswith(".json")]
        for file in json_files:
            file_path = os.path.join(dataset_folder, file)
            try:
                with open(file_path, 'r') as jf:
                    item = json.load(jf)
                if 'id' not in item:
                    item['id'] = os.path.splitext(file)[0]
                mmimdb_data.append(item)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        id_to_genres = {str(item['id']): item['genres'] for item in mmimdb_data}
        json_ids = set(id_to_genres.keys())

        common_feature_ids = text_csv_ids & image_csv_ids & json_ids

        self.filtered_items = []  # Store tuples of (id, text_used, image_used)
        for item_id in common_feature_ids:
            text_available_for_split = item_id in self.text_ids_set
            image_available_for_split = item_id in self.image_ids_set

            # Include item if EITHER text OR image is available according to the lists from load_examples
            if text_available_for_split or image_available_for_split:
                self.filtered_items.append((item_id, text_available_for_split, image_available_for_split))

        if not self.filtered_items:
            # Keep warning, allows execution to continue with empty dataset if needed
            logger.warning(f"ClipMMIMDBDataset '{split}': No items left after filtering. Dataset will be empty.")
            self.valid_ids = np.array([])
            self.item_text_used = []
            self.item_image_used = []
            self.text_features = np.array([])
            self.image_features = np.array([])
            self.item_genres = []
        else:
            self.valid_ids = np.array([item[0] for item in self.filtered_items])
            self.item_text_used = [item[1] for item in self.filtered_items]
            self.item_image_used = [item[2] for item in self.filtered_items]

            self.text_features = np.array([list(text_features_dict[id_].values()) for id_ in self.valid_ids], dtype=np.float32)
            self.image_features = np.array([list(image_features_dict[id_].values()) for id_ in self.valid_ids], dtype=np.float32)
            self.item_genres = [id_to_genres[id_] for id_ in self.valid_ids]

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        text_feat = torch.tensor(self.text_features[idx], dtype=torch.float32)
        image_feat = torch.tensor(self.image_features[idx], dtype=torch.float32)
        target = torch.zeros(self.n_classes, dtype=torch.float32)
        label_idxs = [self.labels.index(genre) for genre in self.item_genres[idx] if genre in self.labels]
        target[label_idxs] = 1.0

        text_used_flag = self.item_text_used[idx]
        image_used_flag = self.item_image_used[idx]

        return {
            "input_ids": text_feat,
            "attention_mask": None,
            "pixel_values": image_feat,
            "label": target,
            "text_used": text_used_flag,
            "image_used": image_used_flag
        }


class SVMClassifier:
    """SVM classifier for multi-label classification using the MultiOutputClassifier approach."""

    def __init__(self, args, num_labels):
        self.args = args
        self.num_labels = num_labels
        self.max_iter = 200000

        # Create a pipeline with StandardScaler and LinearSVC
        base_estimator = make_pipeline(
            StandardScaler(),
            LinearSVC(
                max_iter=self.max_iter,
                C=1.0,
                class_weight='balanced',
                dual=True
            )
        )

        # Use MultiOutputClassifier
        self.classifier = MultiOutputClassifier(base_estimator, n_jobs=-1)
        logging.info(f"Initialized MultiOutputClassifier SVM with {num_labels} labels")

    def fit(self, X, y):
        """Train the SVM classifier."""
        print(f"Training SVM with {X.shape[0]} samples, {X.shape[1]} features, {self.num_labels} labels")

        # Check feature sparsity
        sparsity = np.sum(X == 0) / (X.shape[0] * X.shape[1])
        print(f"Feature sparsity: {sparsity:.2%}")

        # Train the classifier
        self.classifier.fit(X, y)
        print("SVM training complete")

    def predict(self, X):
        """Generate predictions for the given data."""
        return self.classifier.predict(X)


class SVMFeatureExtractor:
    """
    Helper class for extracting precomputed features (CLIP, BLIP, LLaVa)
    from dataset items for SVM models. Assumes features are provided
    as tensors in the dataset item.
    """

    def __init__(self, dataset, args, tokenizer=None):
        self.dataset = dataset
        self.args = args
        # Determine which precomputed features are being used for logging
        feature_types = []
        if getattr(args, 'use_clip_features', False): feature_types.append("CLIP")
        if getattr(args, 'use_blip_features', False): feature_types.append("BLIP")
        if getattr(args, 'use_llava_features', False): feature_types.append("LLaVa")
        if not feature_types:
            logger.warning("SVMFeatureExtractor initialized without any precomputed feature flags (--use_clip/blip/llava) set in args. This is unexpected for SVM.")
            self.feature_type_str = "Unknown (No flags set!)"
        else:
            self.feature_type_str = "/".join(feature_types)

        print(f"Initializing SVMFeatureExtractor using precomputed {self.feature_type_str} features.")
        print(f"  Text Modality Usage: {args.text_modality_percent}%")
        print(f"  Image Modality Usage: {args.image_modality_percent}%")

    def extract_features(self):
        """Extract features and labels from the dataset."""
        X_list = []
        y_list = []

        if len(self.dataset) == 0:
            logger.warning("SVMFeatureExtractor: Dataset is empty, returning empty arrays.")
            num_labels = getattr(self.dataset, 'num_labels', getattr(self.dataset, 'n_classes', 0))
            return np.array([]).reshape(0, 0), np.array([]).reshape(0, num_labels)

        num_items = len(self.dataset)
        print(f"Starting SVM feature extraction for {num_items} items...")

        extraction_errors = 0

        # Process all items
        for i in range(num_items):
            try:
                item = self.dataset[i]
                features = self._extract_and_combine_precomputed(item)

                if features is not None:
                    X_list.append(features)
                    # Ensure label exists and is a tensor before converting
                    if "label" in item and isinstance(item["label"], torch.Tensor):
                        y_list.append(item["label"].cpu().numpy())
                    else:
                        logger.warning(f"Missing or invalid label for item {i}. Skipping sample.")
                        X_list.pop()
                        extraction_errors += 1
                else:
                    extraction_errors += 1

                if (i + 1) % 1000 == 0 or (i + 1) == num_items:
                    print(f"  Processed {i + 1}/{num_items} items...")

            except Exception as e:
                logger.error(f"Unexpected error processing item {i} in SVMFeatureExtractor: {e}", exc_info=True)
                extraction_errors += 1

        if extraction_errors > 0:
            logger.warning(f"Encountered {extraction_errors} errors during feature extraction/processing.")

        if not X_list or not y_list:
            logger.warning("No features/labels could be extracted from any items.")
            num_labels = getattr(self.dataset, 'num_labels', getattr(self.dataset, 'n_classes', 0))
            return np.array([]).reshape(0, 0), np.array([]).reshape(0, num_labels)

        # Stack all features and labels
        try:
            X = np.vstack(X_list)
            y = np.vstack(y_list)
        except ValueError as e:
            logger.error(f"Failed to stack features/labels: {e}. Check for inconsistent shapes.", exc_info=True)
            # Try to return partial data or empty arrays
            num_labels = getattr(self.dataset, 'num_labels', getattr(self.dataset, 'n_classes', 0))
            return np.array([]).reshape(0, 0), np.array([]).reshape(0, num_labels)

        print(f"\nSVM Feature Extraction Summary:")
        print(f"  Extracted features: shape={X.shape}, dtype={X.dtype}, NaNs={np.isnan(X).sum()}")
        if X.size > 0: print(f"    min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
        print(f"  Labels: shape={y.shape}, dtype={y.dtype}")
        print(f"  Number of successfully processed samples: {X.shape[0]}/{num_items}")
        if y.size > 0: print(f"  Samples with >=1 label: {(y.sum(axis=1) > 0).sum()}/{y.shape[0]}")

        return X, y

    def _extract_and_combine_precomputed(self, item):
        """
        Extracts precomputed text and image features from the item
        and combines them based on args.text/image_modality_percent.
        Assumes item['input_ids'] holds text features and
        item['pixel_values'] holds image features as tensors.
        Handles zero-padding if only one modality is selected via args.
        """
        text_features = None
        image_features = None
        final_features = []

        # Extract Text Features if available in item
        if "input_ids" in item and isinstance(item["input_ids"], torch.Tensor) and item["input_ids"].numel() > 0:
            try:
                text_features = item["input_ids"].cpu().numpy().flatten()
                if np.isnan(text_features).any():
                    logger.warning("NaN found in text features for an item. Replacing with zeros.")
                    text_features = np.nan_to_num(text_features)
            except Exception as e:
                logger.warning(f"Could not process text features tensor: {e}")
                text_features = None

        # Extract Image Features if available in item
        if "pixel_values" in item and isinstance(item["pixel_values"], torch.Tensor) and item["pixel_values"].numel() > 0:
            try:
                image_features = item["pixel_values"].cpu().numpy().flatten()
                if np.isnan(image_features).any():
                    logger.warning("NaN found in image features for an item. Replacing with zeros.")
                    image_features = np.nan_to_num(image_features)
            except Exception as e:
                logger.warning(f"Could not process image features tensor: {e}")
                image_features = None

        use_text = self.args.text_modality_percent > 0
        use_image = self.args.image_modality_percent > 0

        if use_text:
            if text_features is not None:
                final_features.append(text_features)
            else:
                logger.warning(f"Text modality requested ({self.args.text_modality_percent}%) but text features are missing or invalid for an item. Cannot proceed for this item.")
                return None

        if use_image:
            if image_features is not None:
                final_features.append(image_features)
            else:
                logger.warning(f"Image modality requested ({self.args.image_modality_percent}%) but image features are missing or invalid for an item. Cannot proceed for this item.")
                return None

        # Handle zero-padding if only one modality *selected* but *both* available
        # This ensures consistent feature vector length for the SVM
        if use_text and not use_image and text_features is not None and image_features is not None:
            # Only text used, pad for image
            final_features.append(np.zeros_like(image_features))
        elif not use_text and use_image and text_features is not None and image_features is not None:
            # Only image used, pad for text - need to insert at beginning
            final_features.insert(0, np.zeros_like(text_features))

        if not final_features:
            logger.warning(f"No features selected based on modality percentages ({self.args.text_modality_percent}% text, {self.args.image_modality_percent}% image) or features missing for an item.")
            return None

        try:
            # Concatenate along axis 0
            combined = np.concatenate(final_features, axis=0)
            return combined
        except ValueError as e:
            # This might happen if zero-padding failed or shapes are unexpectedly different
            logger.error(f"Error concatenating features for item: {e}. Feature shapes: {[f.shape for f in final_features if f is not None]}")
            return None


class CombinedMusic4AllDataset(Dataset):
    # Static cache to store genre lists across instances
    _cached_all_genres = None
    _cached_first_genres = None
    _debug_mode_active = False
    _first_genre_counts = None
    _min_genre_frequency = 3

    def __init__(self, data_dir, text_mbids, image_mbids, tokenizer, transforms, max_seq_length, model_type, split='train', debug_single_genre=False):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transforms = transforms
        self.split = split
        self.model_type = model_type
        self.debug_single_genre = debug_single_genre

        # Set class-level debug mode flag
        CombinedMusic4AllDataset._debug_mode_active = debug_single_genre

        logging.info(f"Initializing CombinedMusic4AllDataset with split={split}, debug_single_genre={debug_single_genre}")

        # Process and separate the prefixed MBIDs
        self.artist_text_mbids = [mbid.replace("artist_", "") for mbid in text_mbids if mbid.startswith("artist_")]
        self.album_text_mbids = [mbid.replace("album_", "") for mbid in text_mbids if mbid.startswith("album_")]
        self.artist_image_mbids = [mbid.replace("artist_", "") for mbid in image_mbids if mbid.startswith("artist_")]
        self.album_image_mbids = [mbid.replace("album_", "") for mbid in image_mbids if mbid.startswith("album_")]

        logging.info(f"Processed {len(self.artist_text_mbids)} artist text MBIDs and {len(self.album_text_mbids)} album text MBIDs")
        logging.info(f"Processed {len(self.artist_image_mbids)} artist image MBIDs and {len(self.album_image_mbids)} album image MBIDs")

        # Load artist and album data directories
        self.artist_data_dir = os.path.join(data_dir, "artist")
        self.album_data_dir = os.path.join(data_dir, "album")

        # Load the data
        self.data = self.load_data()

        # Handle genre lists with caching for consistency
        if debug_single_genre:
            # Using first genres only
            if CombinedMusic4AllDataset._cached_first_genres is None:
                # Count all first genres across the dataset to find those that appear at least 3 times
                if CombinedMusic4AllDataset._first_genre_counts is None:
                    CombinedMusic4AllDataset._first_genre_counts = self.count_first_genres()
                    logging.info(f"Counted {len(CombinedMusic4AllDataset._first_genre_counts)} unique first genres")

                # Get only genres that appear at least MIN_GENRE_FREQUENCY times
                frequent_first_genres = [genre for genre, count in CombinedMusic4AllDataset._first_genre_counts.items()
                                         if count >= CombinedMusic4AllDataset._min_genre_frequency]

                logging.info(f"After filtering (min frequency = {CombinedMusic4AllDataset._min_genre_frequency}): {len(frequent_first_genres)} genres")

                # Sort alphabetically for consistency
                CombinedMusic4AllDataset._cached_first_genres = sorted(frequent_first_genres)
                logging.info(f"DEBUG SINGLE GENRE MODE: Initialized first genres cache with {len(CombinedMusic4AllDataset._cached_first_genres)} frequent genres")
                self.genres = CombinedMusic4AllDataset._cached_first_genres
            else:
                self.genres = CombinedMusic4AllDataset._cached_first_genres
                logging.info(f"DEBUG SINGLE GENRE MODE: Using cached first genres list with {len(self.genres)} genres")
        else:
            # Using all genres
            if CombinedMusic4AllDataset._cached_all_genres is None:

                all_genres = self.get_all_genres()
                CombinedMusic4AllDataset._cached_all_genres = all_genres
                logging.info(f"Initialized all genres cache with {len(all_genres)} genres")
                self.genres = all_genres
            else:
                self.genres = CombinedMusic4AllDataset._cached_all_genres
                logging.info(f"Using cached all genres list with {len(self.genres)} genres")

        # Create genre index mapping
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        self.num_labels = len(self.genres)

        # Filter out items that don't have valid genres
        if debug_single_genre:
            self.data = self.filter_items_by_valid_genres()
            logging.info(f"After filtering out items without valid genres: {len(self.data)} items remain")

        # Log dataset information
        logging.info(f"Loaded {len(self.data)} valid albums/artists")
        logging.info(f"Using {self.num_labels} genre labels")

        # Set up image transforms based on model type
        if model_type == "vilt":
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
            self.image_transforms = get_vilt_image_transforms()
        elif model_type == "mmbt" or model_type == "singlebranch" or model_type == "svm":
            self.image_transforms = get_image_transforms()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def filter_items_by_valid_genres(self):
        """Filter out items that don't have valid genres in the genre index."""
        filtered_data = []

        for item in self.data:
            if item['source_type'] == 'artist':
                genres = item['artist_info']['artist']['genres']
            else:
                genres = item['album_info']['album']['genres']

            if self.debug_single_genre:
                if genres and genres[0] in self.genre_to_idx:
                    filtered_data.append(item)
            else:
                if any(genre in self.genre_to_idx for genre in genres):
                    filtered_data.append(item)

        removed_count = len(self.data) - len(filtered_data)
        if removed_count > 0:
            logging.info(f"Removed {removed_count} items without valid genres")

        return filtered_data

    def count_first_genres(self):
        """Count the frequency of each genre appearing as the first genre."""
        first_genre_counter = Counter()

        for item in self.data:
            if item['source_type'] == 'artist':
                genres = item['artist_info']['artist']['genres']
                if genres:
                    first_genre_counter[genres[0]] += 1
            else:
                genres = item['album_info']['album']['genres']
                if genres:
                    first_genre_counter[genres[0]] += 1

        return first_genre_counter

    def get_all_genres(self):
        """Get all unique genres from both artist and album data."""
        all_genres = set()

        # Process artist data
        for filename in os.listdir(self.artist_data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.artist_data_dir, filename), 'r') as f:
                    data = json.load(f)
                    if 'artist_info' in data:
                        all_genres.update(data['artist_info']['artist']['genres'])

        # Process album data
        for filename in os.listdir(self.album_data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.album_data_dir, filename), 'r') as f:
                    data = json.load(f)
                    if 'album_info' in data:
                        all_genres.update(data['album_info']['album']['genres'])

        return sorted(list(all_genres))

    def load_data(self):
        """Load and combine data from both artist and album sources."""
        combined_data = []

        # Process artist data
        artist_files = 0
        used_artist_files = 0

        for filename in os.listdir(self.artist_data_dir):
            if filename.endswith('.json'):
                artist_files += 1
                try:
                    with open(os.path.join(self.artist_data_dir, filename), 'r') as f:
                        item = json.load(f)
                        text_used = item['mbid'] in self.artist_text_mbids
                        image_used = item['mbid'] in self.artist_image_mbids
                        if not text_used and not image_used:
                            continue

                        # Skip items with empty genres
                        if not item['artist_info']['artist']['genres']:
                            continue

                        used_artist_files += 1
                        item['text_used'] = text_used
                        item['image_used'] = image_used
                        item['source_type'] = 'artist'
                        combined_data.append(item)
                except Exception as e:
                    logging.error(f"Error loading artist file {filename}: {e}")

        # Process album data
        album_files = 0
        used_album_files = 0

        for filename in os.listdir(self.album_data_dir):
            if filename.endswith('.json'):
                album_files += 1
                try:
                    with open(os.path.join(self.album_data_dir, filename), 'r') as f:
                        item = json.load(f)
                        text_used = item['mbid'] in self.album_text_mbids
                        image_used = item['mbid'] in self.album_image_mbids
                        if not text_used and not image_used:
                            continue

                        # Skip items with empty genres
                        if not item['album_info']['album']['genres']:
                            continue

                        used_album_files += 1
                        item['text_used'] = text_used
                        item['image_used'] = image_used
                        item['source_type'] = 'album'
                        combined_data.append(item)
                except Exception as e:
                    logging.error(f"Error loading album file {filename}: {e}")

        logging.info(f"Loaded {used_artist_files}/{artist_files} artist items for split '{self.split}'")
        logging.info(f"Loaded {used_album_files}/{album_files} album items for split '{self.split}'")
        logging.info(f"Total combined items: {len(combined_data)}")

        return combined_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        mbid = item['mbid']
        source_type = item['source_type']

        # Get the appropriate data directory
        data_dir = self.artist_data_dir if source_type == 'artist' else self.album_data_dir

        if source_type == 'artist':
            text = item['artist_info']['artist']['wiki']['summary'] if item['text_used'] else ""
            raw_genres = item['artist_info']['artist']['genres']
        else:
            text = item['album_info']['album']['wiki']['summary'] if item['text_used'] else ""
            raw_genres = item['album_info']['album']['genres']

        # Process genres based on debug mode
        if self.debug_single_genre:
            genres = [raw_genres[0]]
        else:
            genres = raw_genres

        # Create label tensor
        label = torch.zeros(self.num_labels)
        for genre in genres:
            if genre in self.genre_to_idx:
                label[self.genre_to_idx[genre]] = 1

        image = self.load_image(mbid, data_dir) if item['image_used'] else self.transforms(self.get_blank_image())

        if self.model_type == "mmbt":
            result = self.prepare_mmbt_item(text, image, label)
        else:
            result = self.prepare_vilt_singlebranch_item(text, image, label)

        result["text_used"] = item['text_used']
        result["image_used"] = item['image_used']
        return result

    def prepare_mmbt_item(self, text, image, label):
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            "image_start_token": input_ids[0],
            "image_end_token": input_ids[-1],
            "sentence": input_ids[1:-1],
            "attention_mask": attention_mask,
            "image": image,
            "label": label,
        }

    def prepare_vilt_singlebranch_item(self, text, image, label):
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": image,
            "label": label,
        }

    def load_image(self, mbid, data_dir):
        image_files = [f for f in os.listdir(data_dir) if f.startswith(mbid) and f.endswith('.jpg')]

        if not image_files:
            return self.transforms(self.get_blank_image())

        image_path = os.path.join(data_dir, image_files[0])
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transforms(image)
        except (OSError, UnidentifiedImageError) as e:
            print(f"Warning: Corrupted image file encountered: {image_path}", e)
            return self.transforms(self.get_blank_image())

    def get_blank_image(self):
        if self.model_type == "vilt":
            return Image.new('RGB', (384, 384), color='black')
        else:
            return Image.new('RGB', (224, 224), color='black')

    def get_label_frequencies(self):
        """Get frequency of each genre label in the dataset."""
        label_freqs = Counter()

        for row in self.data:
            if row['source_type'] == 'artist':
                genres = row['artist_info']['artist']['genres']
            else:
                genres = row['album_info']['album']['genres']

            if self.debug_single_genre:
                # only count first genre
                if genres:
                    label_freqs[genres[0]] += 1
            else:
                # Normal mode, count all valid genres
                for genre in genres:
                    if genre in self.genre_to_idx:
                        label_freqs[genre] += 1

        return label_freqs


class ClipMusic4AllDataset(Dataset):
    """
    Dataset for Music4All using pre-computed CLIP features stored in CSV files.
    Handles both 'artist' and 'album' data types.
    """

    def __init__(self, data_dir, music4all_data_type, clip_features_dir_text, clip_features_dir_image, text_mbids, image_mbids, labels, split='train'):
        """
        Args:
            data_dir (str): Base data directory (e.g., /opt/datasets). Contains the mmimdb/Music4All_new_modalities structure.
            music4all_data_type (str): 'artist' or 'album'.
            clip_features_dir_text (str): Path to the directory containing 'clip_{type}_encoded_texts' folders.
                                           (e.g., /opt/datasets/mmimdb/Music4All_new_modalities)
            clip_features_dir_image (str): Path to the directory containing 'clip_{type}_encoded_images' folders.
                                           (e.g., /opt/datasets/mmimdb/Music4All_new_modalities)
            text_mbids (list): List of MBIDs for which text modality is available in this split.
            image_mbids (list): List of MBIDs for which image modality is available in this split.
            labels (list): List of all possible genre labels.
            split (str): 'train' or 'eval'.
        """
        self.labels = labels
        self.genres = self.labels
        self.n_classes = len(labels)
        self.split = split
        self.music4all_data_type = music4all_data_type

        self.text_mbids_set = set(text_mbids)
        self.image_mbids_set = set(image_mbids)

        logger.info(f"Initializing ClipMusic4AllDataset for type '{music4all_data_type}', split '{split}'")

        # Construct Paths
        suffix = "test" if split == 'eval' else "train"
        # Specific data path for JSON files
        specific_data_dir = os.path.join(data_dir, music4all_data_type)
        # Paths for CLIP feature CSVs
        text_feature_subdir = f"clip_{music4all_data_type}_encoded_texts"
        image_feature_subdir = f"clip_{music4all_data_type}_encoded_images"
        text_csv_filename = f"clip_{music4all_data_type}_txt_latent_{suffix}.csv"
        image_csv_filename = f"clip_{music4all_data_type}_images_latent_{suffix}.csv"

        text_csv_path = os.path.join(clip_features_dir_text, text_feature_subdir, text_csv_filename)
        image_csv_path = os.path.join(clip_features_dir_image, image_feature_subdir, image_csv_filename)

        logger.info(f"Looking for text features: {text_csv_path}")
        logger.info(f"Looking for image features: {image_csv_path}")
        logger.info(f"Looking for JSON metadata in: {specific_data_dir}")

        # Load CLIP CSV Files
        try:
            text_csv = pd.read_csv(text_csv_path, dtype={'item_id': str})
            image_csv = pd.read_csv(image_csv_path, dtype={'item_id': str})
            logger.info(f"Loaded text CSV: {text_csv.shape}")
            logger.info(f"Loaded image CSV: {image_csv.shape}")
            text_csv_ids = set(text_csv['item_id'])
            image_csv_ids = set(image_csv['item_id'])
            # Store features temporarily keyed by ID for easy lookup after filtering
            text_features_dict = text_csv.set_index('item_id').iloc[:, :].to_dict('index')
            image_features_dict = image_csv.set_index('item_id').iloc[:, :].to_dict('index')
        except FileNotFoundError as e:
            logger.error(f"Error loading CLIP CSV features: {e}")
            raise e
        except KeyError:
            logger.error(f"CSV files must contain an 'item_id' column.")
            raise

        # Load JSON Metadata (Genres)
        id_to_genres = {}
        try:
            json_files = sorted([f for f in os.listdir(specific_data_dir) if f.endswith(".json")])
            logger.info(f"Found {len(json_files)} JSON files in {specific_data_dir}")
            for file in json_files:
                file_path = os.path.join(specific_data_dir, file)
                item_id = os.path.splitext(file)[0]
                try:
                    with open(file_path, 'r') as jf:
                        item_data = json.load(jf)
                    # Extract genres based on type
                    if music4all_data_type == 'artist' and 'artist_info' in item_data:
                        genres = item_data['artist_info']['artist'].get('genres', [])
                    elif music4all_data_type == 'album' and 'album_info' in item_data:
                        genres = item_data['album_info']['album'].get('genres', [])
                    else:
                        genres = []
                    if genres:  # Only include items with genres
                        id_to_genres[item_id] = genres
                except Exception as e_inner:
                    logger.warning(f"Skipping JSON file {file_path} due to error: {e_inner}")
        except FileNotFoundError:
            logger.error(f"JSON metadata directory not found: {specific_data_dir}")
            raise
        logger.info(f"Loaded genre metadata for {len(id_to_genres)} items.")

        json_ids = set(id_to_genres.keys())

        common_feature_ids = text_csv_ids & image_csv_ids & json_ids
        sorted_common_ids = sorted(list(common_feature_ids))
        self.filtered_items = []
        for mbid in sorted_common_ids:
            text_available_for_split = mbid in self.text_mbids_set
            image_available_for_split = mbid in self.image_mbids_set

            # Include item if EITHER text OR image is available according to the lists from load_examples
            if text_available_for_split or image_available_for_split:
                self.filtered_items.append((mbid, text_available_for_split, image_available_for_split))

        if not self.filtered_items:
            logger.warning(f"ClipMusic4AllDataset '{music4all_data_type}/{split}': No items left after filtering. Dataset will be empty.")
            self.valid_ids = np.array([])
            self.item_text_used = []
            self.item_image_used = []
            self.text_features = np.array([])
            self.image_features = np.array([])
            self.item_genres = []
        else:
            self.valid_ids = np.array([item[0] for item in self.filtered_items])
            self.item_text_used = [item[1] for item in self.filtered_items]
            self.item_image_used = [item[2] for item in self.filtered_items]

            self.text_features = np.array([list(text_features_dict[id_].values()) for id_ in self.valid_ids], dtype=np.float32)
            self.image_features = np.array([list(image_features_dict[id_].values()) for id_ in self.valid_ids], dtype=np.float32)
            self.item_genres = [id_to_genres[id_] for id_ in self.valid_ids]

        self.num_labels = self.n_classes

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        if idx >= len(self.valid_ids):
            raise IndexError("Index out of bounds")

        # Add text_used/image_used flags to the returned dictionary
        text_feat = torch.tensor(self.text_features[idx], dtype=torch.float32)
        image_feat = torch.tensor(self.image_features[idx], dtype=torch.float32)

        target = torch.zeros(self.n_classes, dtype=torch.float32)
        current_genres = self.item_genres[idx]
        label_idxs = [self.labels.index(genre) for genre in current_genres if genre in self.labels]
        if label_idxs:
            target[label_idxs] = 1.0

        text_used_flag = self.item_text_used[idx]
        image_used_flag = self.item_image_used[idx]

        return {
            "input_ids": text_feat,
            "attention_mask": None,
            "pixel_values": image_feat,
            "label": target,
            "text_used": text_used_flag,
            "image_used": image_used_flag
        }

    def get_label_frequencies(self):
        """Calculate frequency of each genre label present in the filtered dataset."""
        label_freqs = Counter()
        valid_genres_in_dataset = set(self.labels)  # Use the provided full label list
        for genres in self.item_genres:
            # Count only genres that are in the official label list
            valid_item_genres = [g for g in genres if g in valid_genres_in_dataset]
            label_freqs.update(valid_item_genres)
        return label_freqs

    def get_all_genres(self):
        """Returns the predefined list of labels passed during initialization."""
        return self.labels
