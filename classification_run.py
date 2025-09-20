import argparse
import glob
import json
import logging
import math
import os
import random
from collections import defaultdict, Counter
import sys

import wandb

# Parse the --cuda_visible_devices argument
early_parser = argparse.ArgumentParser()
early_parser.add_argument("--cuda_visible_devices", type=str, default=None,
                          help="Set CUDA_VISIBLE_DEVICES environment variable")
args, unknown = early_parser.parse_known_args()

# Set the environment variable
if args.cuda_visible_devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

# Remove the parsed argument from sys.argv
sys.argv = [sys.argv[0]] + unknown

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange
from classification_utils import (ImageEncoder, JsonlDataset, collate_fn, get_image_transforms, get_mmimdb_labels,
                                  Custom_MMBTForClassification, Music4AllDataset, Custom_ViLTForClassification,
                                  get_vilt_image_transforms, SingleBranchModel, LLaVaMMIMDBDataset, BlipMMIMDBDataset,
                                  SVMClassifier, SVMFeatureExtractor, CombinedMusic4AllDataset, ClipMMIMDBDataset, ClipMusic4AllDataset)

import transformers
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    get_linear_schedule_with_warmup, MMBTForClassification,
    ViltConfig, ViltForImagesAndTextClassification, ViltProcessor
)
from transformers.trainer_utils import is_main_process
from datetime import datetime


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if args.dataset == 'music4all':
        # Get label frequencies
        label_freqs = train_dataset.get_label_frequencies()

        # Calculate inverse class weights
        weights = torch.ones(train_dataset.num_labels, device=args.device)

        # Check if genre_to_idx exists
        if hasattr(train_dataset, 'genre_to_idx'):
            for genre, idx in train_dataset.genre_to_idx.items():
                if genre in label_freqs and label_freqs[genre] > 0:
                    weights[idx] = 1.0 / math.log(1.2 + label_freqs[genre])
        else:
            # For SMOTEAugmentedDataset, use index directly
            for i in range(train_dataset.num_labels):
                genre = train_dataset.genres[i]
                if genre in label_freqs and label_freqs[genre] > 0:
                    weights[i] = 1.0 / math.log(1.2 + label_freqs[genre])

        # Normalize weights
        weights = weights / weights.mean()

        criterion = lambda logits, targets: torch.mean(
            weights.unsqueeze(0) * torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps > 0:
        warmup_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        scheduler = warmup_scheduler
    else:
        # If no warmup, just use cosine annealing
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=int(t_total),
            pct_start=0.3,  # Spend 30% of time warming up
            anneal_strategy='cos'
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    if args.local_rank != -1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_f1, n_no_improve = 0, 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) if (t is not None and isinstance(t, torch.Tensor)) else t for t in batch)

            if args.model_type == "mmbt":
                labels = batch[5]
                text_used = batch[6]
                image_used = batch[7]
                inputs = {
                    "input_ids": batch[0],
                    "input_modal": batch[2],
                    "attention_mask": batch[1],
                    "modal_start_tokens": batch[3],
                    "modal_end_tokens": batch[4],
                    "return_dict": False,
                    "text_used": text_used, 
                    "image_used": image_used
                }
            elif args.model_type == "vilt":
                labels = batch[3]
                text_used = batch[4]
                image_used = batch[5]
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "pixel_values": batch[2],
                    "text_used": text_used, 
                    "image_used": image_used
                }
            elif args.model_type == "singlebranch":
                    labels = batch[3]
                    text_used = batch[4]
                    image_used = batch[5]
                    # Always provide both modalities; missing modalities are zeroed out by collate_fn
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "pixel_values": batch[2],
                        "text_used": text_used,
                        "image_used": image_used
                    }

            outputs = model(**inputs)
            logits = outputs[0]
            loss = criterion(logits, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                #learning_rate_scalar = scheduler.get_last_lr()[0]
                scheduler.step()
                learning_rate_scalar = optimizer.param_groups[0]['lr']
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, criterion)
                        for key, value in results.items():
                            eval_key = f"eval_{key}"
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    # print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.save_model:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                if args.use_wandb:
                    log_wandb = {"train_f1": f1_score(labels.detach().cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy() > 0.5, average="samples", zero_division=0),
                                 "train_loss": (tr_loss - logging_loss) / args.logging_steps}

                    wandb.log(log_wandb)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.local_rank == -1:
            results = evaluate(args, model, tokenizer, criterion)
            if results["Validation F1"] > best_f1:
                best_f1 = results["Validation F1"]
                n_no_improve = 0
            else:
                n_no_improve += 1

            if n_no_improve > args.patience:
                train_iterator.close()
                break

            if args.use_wandb:
                wandb.log({"learning_rate": float(learning_rate_scalar)})

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, criterion, prefix=""):
    eval_output_dir = args.output_dir
    eval_dataset = load_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn, drop_last=False, pin_memory=True)

    if args.n_gpu > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            batch = tuple(t.to(args.device) if (t is not None and isinstance(t, torch.Tensor)) else t for t in batch)
            if args.model_type == "mmbt":
                labels = batch[5]
                text_used = batch[6]
                image_used = batch[7]
                inputs = {
                    "input_ids": batch[0],
                    "input_modal": batch[2] if args.image_modality_percent > 0 else None,
                    "attention_mask": batch[1],
                    "modal_start_tokens": batch[3],
                    "modal_end_tokens": batch[4],
                    "return_dict": False,
                    "text_used": text_used,
                    "image_used": image_used
                }
            elif args.model_type == "vilt" or args.model_type == "singlebranch":
                labels = batch[3]
                text_used = batch[4]
                image_used = batch[5]
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "pixel_values": batch[2],
                    "text_used": text_used,
                    "image_used": image_used
                }

            outputs = model(**inputs)
            logits = outputs[0]
            tmp_eval_loss = criterion(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > 0.5, axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps



    all_item_ids_in_eval_order = []
    if hasattr(eval_dataset, 'valid_ids'):  # For precomputed datasets
        all_item_ids_in_eval_order = [str(id_val) for id_val in eval_dataset.valid_ids]
    elif hasattr(eval_dataset, 'data'):
        # Ensure 'mbid' or 'id' is present and consistent
        id_key = 'mbid' if args.dataset == 'music4all' else 'id'
        all_item_ids_in_eval_order = [str(item[id_key]) for item in eval_dataset.data]
    else:
        logger.warning("Could not determine how to get item IDs from eval_dataset for saving.")

    if all_item_ids_in_eval_order and preds is not None and out_label_ids is not None:
        # Ensure lengths match
        if not (len(all_item_ids_in_eval_order) == preds.shape[0] == out_label_ids.shape[0]):
            logger.warning(f"Mismatch in lengths for saving predictions: "
                           f"IDs={len(all_item_ids_in_eval_order)}, Preds={preds.shape[0]}, GT={out_label_ids.shape[0]}. "
                           f"Skipping saving predictions for qualitative analysis.")
        else:
            predictions_save_path = os.path.join(eval_output_dir, f"{prefix}qualitative_preds_gt.npz")
            try:
                np.savez_compressed(
                    predictions_save_path,
                    item_ids=np.array(all_item_ids_in_eval_order, dtype=object),  # Save as object array for strings
                    predictions=preds,  # boolean array
                    ground_truths=out_label_ids.astype(bool)  # boolean array
                )
                logger.info(f"Saved predictions and ground truths for qualitative analysis to: {predictions_save_path}")
            except Exception as e:
                logger.error(f"Failed to save qualitative predictions: {e}")
    else:
        logger.warning("Not saving qualitative predictions because some data is missing (item_ids, preds, or out_label_ids).")


    result = {
        "Validation Loss": eval_loss,
        "Validation F1": f1_score(out_label_ids, preds, average="samples", zero_division=0),
        "Validation Precision": precision_score(out_label_ids, preds, average="samples", zero_division=0),
        "Validation Recall": recall_score(out_label_ids, preds, average="samples", zero_division=0),
        "Validation Hamming Loss": hamming_loss(out_label_ids, preds)
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if args.use_wandb:
        wandb.log(result)

    return result


def load_examples(args, tokenizer, evaluate=False, labels_override=None):
    using_precomputed = args.use_llava_features or args.use_blip_features or args.use_clip_features
    logger.info(f"Loading examples for {'evaluation' if evaluate else 'training'}. Model type: {args.model_type}. Precomputed features used: {using_precomputed}")
    logger.info(f"Train on single modality only mode: {args.train_on_single_modality_only}")

    text_ids, image_ids = [], []
    text_mbids, image_mbids = [], []

    # Helper function for Music4All NN model MBID loading logic
    def _get_music4all_nn_mbids(split_content, type_name_logging, is_eval_flag, args_local):
        _text_mbids_list, _image_mbids_list = [], []

        if is_eval_flag:  # Evaluation always uses percentages
            text_lookup_key = str(args_local.text_modality_percent)
            image_lookup_key = str(args_local.image_modality_percent)
            logger.info(f"NN Evaluation ({type_name_logging}): Loading IDs based on percentages from 'missing_modality': Text {args_local.text_modality_percent}%, Image {args_local.image_modality_percent}%")
            if 'missing_modality' not in split_content: raise KeyError(f"'missing_modality' key not found for {type_name_logging} in split_content")
            _text_mbids_list = split_content['missing_modality'].get(text_lookup_key, [])
            _image_mbids_list = split_content['missing_modality'].get(image_lookup_key, [])
        else:  # Training data loading for NN models
            full_train_mbids_from_split = split_content.get('train', [])
            if not full_train_mbids_from_split: logger.warning(f"NN Training ({type_name_logging}): 'train' key missing or empty in split_content.")

            if args_local.train_on_single_modality_only:
                logger.info(f"NN Training ({type_name_logging}) in train_on_single_modality_only mode.")
                if args_local.text_modality_percent == 0 and args_local.image_modality_percent > 0:  # Image-only
                    logger.info(f"Configuring for IMAGE-ONLY training ({type_name_logging}).")
                    _text_mbids_list = []
                    _image_mbids_list = full_train_mbids_from_split
                elif args_local.image_modality_percent == 0 and args_local.text_modality_percent > 0:  # Text-only
                    logger.info(f"Configuring for TEXT-ONLY training ({type_name_logging}).")
                    _text_mbids_list = full_train_mbids_from_split
                    _image_mbids_list = []
                elif args_local.text_modality_percent == 0 and args_local.image_modality_percent == 0:
                    logger.warning(f"NN Training ({type_name_logging}): Both percentages are 0 in train_on_single_modality_only mode. No data.")
                    _text_mbids_list, _image_mbids_list = [], []
                else:  # Both > 0, train bi-modally
                    logger.info(f"NN Training ({type_name_logging}): Both percentages > 0 in train_on_single_modality_only. Configuring for BI-MODAL training using full 'train' split.")
                    _text_mbids_list = full_train_mbids_from_split
                    _image_mbids_list = full_train_mbids_from_split
            else:  # Standard mode: train bi-modally using full 'train' split
                logger.info(f"NN Training ({type_name_logging}) in standard mode. Percentages ignored for training data composition, using full 'train' split for bi-modal training.")
                _text_mbids_list = full_train_mbids_from_split
                _image_mbids_list = full_train_mbids_from_split
        return _text_mbids_list, _image_mbids_list


    if args.dataset == 'mmimdb':
        # Load the modality splits file
        modality_split_file = os.path.join(args.data_dir, args.dataset, f"{args.dataset}_modality_splits.json")
        with open(modality_split_file, 'r') as f:
            splits = json.load(f)

        if args.model_type == 'svm':
            # SVM: Ignore modality percentages at loading. Load the 'train' set for training,
            # and the '100%' set for evaluation (representing the full test set).
            # The SVMFeatureExtractor handles modality usage later.
            if evaluate:
                split_key = '100'
                logger.info("SVM Evaluation: Loading IDs based on '100%' availability key from 'missing_modality'.")
                # Load the same list for both, as SVM feature extractor handles usage %
                text_ids = splits['missing_modality'][split_key]
                image_ids = splits['missing_modality'][split_key]
            else:  # SVM Training
                split_key = 'train'
                logger.info("SVM Training: Loading IDs based on 'train' split key.")

                # Load the same list for both
                text_ids = splits[split_key]
                image_ids = splits[split_key]
            logger.info(f"SVM: Loaded {len(text_ids)} IDs for {split_key} split.")

        else:  # Neural Network Models (MMBT, ViLT, SingleBranch)
            if evaluate:
                # NN Evaluation: Use modality percentages
                text_lookup_key = str(args.text_modality_percent)
                image_lookup_key = str(args.image_modality_percent)
                logger.info(f"NN Evaluation: Loading IDs based on percentages: Text {args.text_modality_percent}% (Key: {text_lookup_key}), Image {args.image_modality_percent}% (Key: {image_lookup_key})")
                if 'missing_modality' not in splits: raise KeyError(f"'missing_modality' key not found in {modality_split_file}")

                text_ids = splits['missing_modality'].get(text_lookup_key, [])
                image_ids = splits['missing_modality'].get(image_lookup_key, [])
                if not text_ids: logger.warning(f"Text lookup key '{text_lookup_key}' not found or empty in 'missing_modality'.")
                if not image_ids: logger.warning(f"Image lookup key '{image_lookup_key}' not found or empty in 'missing_modality'.")
            else:  # NN Training (MMIMDB)
                split_key = 'train'
                full_train_ids = splits[split_key]

                if args.train_on_single_modality_only:
                    logger.info("NN Training (MMIMDB) in train_on_single_modality_only mode.")
                    if args.text_modality_percent == 0 and args.image_modality_percent > 0:  # Image-only
                        logger.info("Configuring for IMAGE-ONLY training (MMIMDB).")
                        text_ids = []
                        image_ids = full_train_ids
                    elif args.image_modality_percent == 0 and args.text_modality_percent > 0:  # Text-only
                        logger.info("Configuring for TEXT-ONLY training (MMIMDB).")
                        text_ids = full_train_ids
                        image_ids = []
                    elif args.text_modality_percent == 0 and args.image_modality_percent == 0:
                        logger.warning("NN Training (MMIMDB): Both percentages are 0 in train_on_single_modality_only mode. This will likely lead to no data.")
                        text_ids, image_ids = [], []
                    else:  # Both > 0, or other edge cases, train bi-modally (or as per percentages if both > 0)
                        logger.info("NN Training (MMIMDB): Both percentages > 0 in train_on_single_modality_only mode. Configuring for BI-MODAL training using full 'train' split.")
                        text_ids = full_train_ids
                        image_ids = full_train_ids
                else:  # Standard mode (train_on_single_modality_only is False)
                    # For NNs, train bi-modally using full 'train' split
                    logger.info("NN Training (MMIMDB) in standard mode (or SVM). Percentages (if NN) usually ignored for training data composition, using full 'train' split for bi-modal NN training.")
                    text_ids = full_train_ids
                    image_ids = full_train_ids

            logger.info(f"NN {'Eval' if evaluate else 'Train'}: Loaded {len(text_ids)} text IDs and {len(image_ids)} image IDs.")

        if labels_override:
            labels = labels_override
            logging.info(f"Overriding labels for dataset construction. Using {len(labels)} labels: {labels}")
        else:
            # Use default MMIMDB labels if no override provided
            labels = get_mmimdb_labels()
            print("labels", labels)
            logging.info(f"Using default {len(labels)} MMIMDB labels.")

        dataset_folder = os.path.join(args.data_dir, args.dataset, "dataset")

        if args.use_llava_features:
            dataset = LLaVaMMIMDBDataset(
                dataset_folder,
                args.llava_features_dir_image,
                args.llava_features_dir_text,
                text_ids,
                image_ids,
                labels,
                split='eval' if evaluate else 'train'
            )
        elif args.use_blip_features:
            dataset = BlipMMIMDBDataset(
                dataset_folder,
                args.blip_features_dir_image,
                args.blip_features_dir_text,
                text_ids,
                image_ids,
                labels,
                split='eval' if evaluate else 'train'
            )
        elif args.use_clip_features:
            dataset = ClipMMIMDBDataset(
                dataset_folder,
                args.clip_features_dir_image,
                args.clip_features_dir_text,
                text_ids,
                image_ids,
                labels,
                split='eval' if evaluate else 'train'
            )
        else:
            transforms = get_image_transforms()
            dataset = JsonlDataset(
                dataset_folder, text_ids, image_ids, tokenizer, transforms, labels,
                args.max_seq_length - args.num_image_embeds - 2,
                args.model_type,
                split='eval' if evaluate else 'train'
            )
    elif args.dataset == 'music4all':
        print("Loading Music4All dataset")
        music4all_base_data_dir = os.path.join(args.data_dir)

        # Special handling for SVM or SingleBranch with CLIP features
        if args.use_clip_features and args.model_type in ['svm', 'singlebranch']:
            logger.info(f"Using ClipMusic4AllDataset for model_type='{args.model_type}' and data_type='{args.music4all_data_type}'")
            if args.music4all_data_type == "Both":
                raise NotImplementedError(f"ClipMusic4AllDataset with 'Both' Music4All types is not yet supported for {args.model_type}. Please specify 'artist' or 'album'.")
            if args.music4all_data_type not in ["artist", "album"]:
                raise ValueError(f"Invalid music4all_data_type '{args.music4all_data_type}' for {args.model_type}+CLIP. Must be 'artist' or 'album'.")

            split_file = os.path.join(music4all_base_data_dir, f"{args.music4all_data_type}_modality_splits.json")
            logger.info(f"Loading split file for MBIDs: {split_file}")
            if not os.path.exists(split_file): raise FileNotFoundError(f"Split file not found: {split_file}")
            with open(split_file, 'r') as f:
                splits = json.load(f)

            # Determine MBIDs based on model type and evaluate flag for CLIP dataset
            if args.model_type == 'svm':
                if evaluate:
                    split_key = '100'
                    logger.info(f"SVM Evaluation ({args.music4all_data_type} CLIP): Loading IDs based on '100%' availability key.")
                    if 'missing_modality' not in splits or split_key not in splits['missing_modality']: raise KeyError(f"Split file {split_file} missing 'missing_modality' or key '{split_key}'")
                    text_mbids = splits['missing_modality'][split_key]
                    image_mbids = splits['missing_modality'][split_key]
                else:  # SVM Training
                    split_key = 'train'
                    logger.info(f"SVM Training ({args.music4all_data_type} CLIP): Loading IDs based on 'train' split key.")
                    if split_key not in splits: raise KeyError(f"Split file {split_file} missing '{split_key}' key")
                    text_mbids = splits[split_key]
                    image_mbids = splits[split_key]
                logger.info(f"SVM CLIP: Loaded {len(text_mbids)} MBIDs for {args.music4all_data_type}.")
            else:  # SingleBranch NN with CLIP
                text_mbids, image_mbids = _get_music4all_nn_mbids(splits, f"{args.music4all_data_type} CLIP NN", evaluate, args)
                logger.info(f"NN CLIP ({'Eval' if evaluate else 'Train'}): Loaded {len(text_mbids)} text MBIDs and {len(image_mbids)} image MBIDs for {args.music4all_data_type}.")

            try:
                temp_dataset_path = os.path.join(music4all_base_data_dir, args.music4all_data_type)
                # Use 'svm' temporarily for model_type as it doesn't affect genre loading in Music4AllDataset
                temp_ds = Music4AllDataset(temp_dataset_path, [], [], None, None, 0, 'svm', args.music4all_data_type, 'train')
                all_labels = temp_ds.genres
                del temp_ds
            except NameError:
                raise RuntimeError("Cannot determine Music4All labels. Ensure Music4AllDataset is defined or provide a label function.")
            except Exception as e:
                logger.error(f"Error getting labels via temp dataset: {e}")
                raise

            if labels_override and args.model_type == 'svm':
                labels = labels_override
                logger.info(f"Overriding labels for ClipMusic4AllDataset (SVM). Using {len(labels)} labels...")
            else:
                labels = all_labels
                logger.info(f"Using default {len(labels)} Music4All labels for ClipMusic4AllDataset.")

            clip_features_dir_text_abs = music4all_base_data_dir
            clip_features_dir_image_abs = music4all_base_data_dir

            dataset = ClipMusic4AllDataset(
                os.path.join(music4all_base_data_dir),
                args.music4all_data_type,
                clip_features_dir_text_abs,
                clip_features_dir_image_abs,
                text_mbids,
                image_mbids,
                labels,
                split='eval' if evaluate else 'train'
            )
            args.num_labels = dataset.num_labels

        elif args.music4all_data_type == "Both":
            artist_split_file = os.path.join(music4all_base_data_dir, "artist_modality_splits.json")
            album_split_file = os.path.join(music4all_base_data_dir, "album_modality_splits.json")

            logging.info(f"Loading artist split file: {artist_split_file}")
            logging.info(f"Loading album split file: {album_split_file}")

            if not os.path.exists(artist_split_file) or not os.path.exists(album_split_file):
                raise FileNotFoundError(f"Split files not found: {artist_split_file} or {album_split_file}")

            with open(artist_split_file, 'r') as f:
                artist_splits = json.load(f)
            with open(album_split_file, 'r') as f:
                album_splits = json.load(f)

            # Determine MBIDs based on model type and evaluate flag for Combined dataset
            if args.model_type == 'svm':
                if evaluate:
                    logger.info("SVM Evaluation (Both): Loading IDs based on '100%' availability key.")
                    if 'missing_modality' not in artist_splits or '100' not in artist_splits['missing_modality']: raise KeyError("Artist split missing 'missing_modality' or key '100'")
                    if 'missing_modality' not in album_splits or '100' not in album_splits['missing_modality']: raise KeyError("Album split missing 'missing_modality' or key '100'")
                    artist_text_mbids = artist_splits['missing_modality']['100']
                    artist_image_mbids = artist_splits['missing_modality']['100']
                    album_text_mbids = album_splits['missing_modality']['100']
                    album_image_mbids = album_splits['missing_modality']['100']
                else:  # SVM Training
                    logger.info("SVM Training (Both): Loading IDs based on 'train' split key.")
                    if 'train' not in artist_splits: raise KeyError("Artist split missing 'train' key")
                    if 'train' not in album_splits: raise KeyError("Album split missing 'train' key")
                    artist_text_mbids = artist_splits['train']
                    artist_image_mbids = artist_splits['train']
                    album_text_mbids = album_splits['train']
                    album_image_mbids = album_splits['train']
            else:  # NN Models for "Both"
                artist_text_mbids_list, artist_image_mbids_list = _get_music4all_nn_mbids(artist_splits, "Artist (for Both)", evaluate, args)
                album_text_mbids_list, album_image_mbids_list = _get_music4all_nn_mbids(album_splits, "Album (for Both)", evaluate, args)

                artist_text_mbids = [f"artist_{mbid}" for mbid in artist_text_mbids_list]
                artist_image_mbids = [f"artist_{mbid}" for mbid in artist_image_mbids_list]
                album_text_mbids = [f"album_{mbid}" for mbid in album_text_mbids_list]
                album_image_mbids = [f"album_{mbid}" for mbid in album_image_mbids_list]


            artist_text_mbids = [f"artist_{mbid}" for mbid in artist_text_mbids]
            artist_image_mbids = [f"artist_{mbid}" for mbid in artist_image_mbids]
            album_text_mbids = [f"album_{mbid}" for mbid in album_text_mbids]
            album_image_mbids = [f"album_{mbid}" for mbid in album_image_mbids]

            text_mbids = artist_text_mbids + album_text_mbids
            image_mbids = artist_image_mbids + album_image_mbids

            logging.info(f"Combined {len(text_mbids)} text MBIDs and {len(image_mbids)} image MBIDs for CombinedMusic4AllDataset")

            dataset = CombinedMusic4AllDataset(
                args.data_dir,
                text_mbids,
                image_mbids,
                tokenizer,
                get_image_transforms(),
                args.max_seq_length - args.num_image_embeds - 2,
                args.model_type,
                split='eval' if evaluate else 'train',
                debug_single_genre=args.debug_single_genre
            )
            args.num_labels = dataset.num_labels

        else:  # Single type handling (artist or album, not SVM/SingleBranch+CLIP)
            if args.music4all_data_type not in ["artist", "album"]:
                raise ValueError(f"Invalid music4all_data_type '{args.music4all_data_type}'. Must be 'artist' or 'album'.")

            split_file = os.path.join(music4all_base_data_dir, f"{args.music4all_data_type}_modality_splits.json")
            logging.info(f"Loading split file: {split_file}")
            if not os.path.exists(split_file): raise FileNotFoundError(f"Split file not found: {split_file}")
            with open(split_file, 'r') as f:
                splits = json.load(f)

            # Determine MBIDs based on model type and evaluate flag for single type dataset
            if args.model_type == 'svm':
                if evaluate:
                    split_key = '100'
                    logger.info(f"SVM Evaluation ({args.music4all_data_type}): Loading IDs based on '100%' availability key.")
                    if 'missing_modality' not in splits or split_key not in splits['missing_modality']: raise KeyError(f"Split file {split_file} missing 'missing_modality' or key '{split_key}'")
                    text_mbids = splits['missing_modality'][split_key]
                    image_mbids = splits['missing_modality'][split_key]
                else:  # SVM Training
                    split_key = 'train'
                    logger.info(f"SVM Training ({args.music4all_data_type}): Loading IDs based on 'train' split key.")
                    if split_key not in splits: raise KeyError(f"Split file {split_file} missing '{split_key}' key")
                    text_mbids = splits[split_key]
                    image_mbids = splits[split_key]
                logger.info(f"SVM: Loaded {len(text_mbids)} MBIDs for {args.music4all_data_type}.")
            else:  # NN Models for single type (artist or album)
                text_mbids, image_mbids = _get_music4all_nn_mbids(splits, str(args.music4all_data_type), evaluate, args)
                logger.info(f"NN {'Eval' if evaluate else 'Train'} ({args.music4all_data_type}): Loaded {len(text_mbids)} text MBIDs and {len(image_mbids)} image MBIDs.")

            logging.info(f"Loaded {len(text_mbids)} text MBIDs and {len(image_mbids)} image MBIDs from split file for Music4AllDataset")

            specific_data_dir = os.path.join(music4all_base_data_dir, args.music4all_data_type)
            logging.info(f"Loading data from directory: {specific_data_dir}")

            dataset = Music4AllDataset(
                specific_data_dir,
                text_mbids,
                image_mbids,
                tokenizer,
                get_image_transforms(),
                args.max_seq_length - args.num_image_embeds - 2,
                args.model_type,
                args.music4all_data_type,
                split='eval' if evaluate else 'train'
            )
            args.num_labels = dataset.num_labels

        logging.info(f"Music4All dataset loaded with {len(dataset)} samples for {'evaluation' if evaluate else 'training'}")
        args.num_labels = dataset.num_labels
        logging.info(f"Number of labels: {args.num_labels}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir."
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory."
    )

    # Other parameters
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--num_image_embeds", default=1, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)  # 8
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)  # 8
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.00, type=float)  # 0.00
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=3, type=int)  # 0
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_all_checkpoints", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--server_ip", type=str, default="")
    parser.add_argument("--server_port", type=str, default="")

    parser.add_argument("--text_modality_percent", type=int, default=100, choices=[0, 10, 30, 50, 70, 90, 100])
    parser.add_argument("--image_modality_percent", type=int, default=100, choices=[0, 10, 30, 50, 70, 90, 100])

    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--dataset", type=str, default="mmimdb", choices=["mmimdb", "music4all"])
    parser.add_argument("--music4all_data_type", type=str, default=None, choices=["album", "artist", "Both", "None"])
    parser.add_argument("--model_type", type=str, default="mmbt", choices=["mmbt", "vilt", "singlebranch", "svm"])
    parser.add_argument("--use_llava_features", action="store_true")
    parser.add_argument("--llava_features_dir_image", type=str, default="/opt/datasets/mmimdb/llava_encoded_images")
    parser.add_argument("--llava_features_dir_text", type=str, default="/opt/datasets/mmimdb/llava_encoded_texts")

    parser.add_argument("--use_blip_features", action="store_true",
                        help="Whether to use BLIP features for singlebranch model on MMIMDB.")
    parser.add_argument("--blip_features_dir_image", type=str,
                        default="/opt/datasets/mmimdb/blip_encoded_images",
                        help="Directory containing BLIP-encoded image features for MMIMDB.")
    parser.add_argument("--blip_features_dir_text", type=str,
                        default="/opt/datasets/mmimdb/blip_encoded_texts",
                        help="Directory containing BLIP-encoded text features for MMIMDB.")

    parser.add_argument("--use_clip_features", action="store_true",
                        help="Whether to use CLIP features for model on MMIMDB.")
    parser.add_argument("--clip_features_dir_image", type=str,
                        default="/opt/datasets/mmimdb/clip_encoded_images",
                        help="Directory containing CLIP-encoded image features for MMIMDB.")
    parser.add_argument("--clip_features_dir_text", type=str,
                        default="/opt/datasets/mmimdb/clip_encoded_texts",
                        help="Directory containing CLIP-encoded text features for MMIMDB.")

    parser.add_argument("--debug_single_genre", action="store_true",
                        help="Debug mode: Use only the first genre for each item (for CombinedMusic4AllDataset)")

    parser.add_argument("--svm_num_classes", type=int, default=None,
                        help="SVM ONLY: Limit training/evaluation to the top N most frequent classes. "
                             "If None, uses all available classes. (Default: None)")
    parser.add_argument("--svm_num_samples", type=int, default=None,
                        help="SVM ONLY: Limit the number of training samples by random sampling "
                             "(after filtering for selected classes). If None, uses all valid samples. (Default: None)")

    parser.add_argument("--train_on_single_modality_only", action="store_true",
                        help="If true, and a modality percent is 0, train NN models strictly on the other modality. "
                             "If false (default), NN training is bi-modal from 'train' split, and percentages affect eval only.")

    args = parser.parse_args()

    if args.use_wandb:
        time = datetime.now().strftime("%m-%d_%H-%M-%S")
        run_name_suffix = f"_single_train" if args.train_on_single_modality_only and (args.text_modality_percent == 0 or args.image_modality_percent == 0) else ""
        if args.dataset=="mmimdb":
            wandb.init(project="2_Multimodal-Genre-Classification", config=args, name=f"MMIMDB_{args.model_type}_img{args.image_modality_percent}_txt{args.text_modality_percent}{run_name_suffix}_{time}")
        elif args.dataset=="music4all":
            wandb.init(project="2_Multimodal-Genre-Classification", config=args, name=f"Music4All_{args.music4all_data_type}_{args.model_type}_img{args.image_modality_percent}_txt{args.text_modality_percent}{run_name_suffix}_{time}")

    if args.dataset == "music4all":
        args.output_dir = args.output_dir + "_" + args.dataset + "_" + args.music4all_data_type + "_image_" + str(args.image_modality_percent) + "_text_" + str(args.text_modality_percent)
    else:
        args.output_dir = args.output_dir + "_" + args.dataset + "_image_" + str(args.image_modality_percent) + "_text_" + str(args.text_modality_percent)

    if args.train_on_single_modality_only and (args.text_modality_percent == 0 or args.image_modality_percent == 0):
        args.output_dir += "_single_train_mode"

    time = datetime.now().strftime("%m-%d_%H-%M-%S")
    args.output_dir = args.output_dir + "_" + time

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir.".format(
                args.output_dir
            )
        )

    if args.music4all_data_type == "None":
        args.music4all_data_type = None

    if args.music4all_data_type is not None and args.dataset == "mmimdb":
        raise ValueError("music4all_data_type can only be used with the music4all dataset")
    if args.music4all_data_type is None and args.dataset == "music4all":
        raise ValueError("music4all_data_type is required for the music4all dataset")

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    print('device:', device, args.local_rank, args.n_gpu)
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )
    tokenizer.truncation_side = "right"

    train_dataset = load_examples(args, tokenizer, evaluate=False)

    if args.dataset == "music4all":
        num_labels = train_dataset.num_labels
        labels = train_dataset.genres
    else:
        labels = get_mmimdb_labels()
        num_labels = len(labels)

    if args.model_type == "mmbt":
        transformer_config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=args.cache_dir,
        )

        transformer = AutoModel.from_pretrained(
            args.model_name_or_path, config=transformer_config, cache_dir=args.cache_dir
        )

        img_encoder = ImageEncoder(args)
        config = MMBTConfig(transformer_config, num_labels=num_labels)
        config.num_labels = num_labels
        model = Custom_MMBTForClassification(config=config, transformer=transformer, encoder=img_encoder)
    elif args.model_type == "vilt":
        config = ViltConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.max_position_embeddings = args.max_seq_length
        config.num_labels = num_labels
        config.num_images = 1
        model = Custom_ViLTForClassification(args, config, num_labels)
    elif args.model_type == "singlebranch":
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        model = SingleBranchModel(args, num_labels)
    elif args.model_type == "svm":
        print("\n" + "=" * 50)
        print("INITIALIZING SVM MODEL")
        print("=" * 50)
        print(f"Dataset: {args.dataset}")
        if args.use_llava_features:
            print("Using: LLaVa features")
        elif args.use_blip_features:
            print("Using: BLIP features")
        elif args.use_clip_features:
            print("Using: CLIP features")
        else:
            print("\nERROR: SVM classifier should be used with pre-computed embeddings.")
            print("Please use --use_llava_features, --use_blip_features, or --use_clip_features.")
            sys.exit(1)  # Exit if no features specified for SVM

        if args.svm_num_classes is not None:
            print(f"\nAttempting to select top {args.svm_num_classes} most frequent classes...")
            # Load temporary full dataset just to get frequencies and all possible labels
            print("Loading preliminary dataset to calculate label frequencies...")
            temp_train_dataset = load_examples(args, tokenizer, evaluate=False, labels_override=None)  # Load with ALL labels first

            if hasattr(temp_train_dataset, 'get_label_frequencies'):
                label_freqs = temp_train_dataset.get_label_frequencies()
                all_possible_labels = temp_train_dataset.genres  # Get all labels defined by the dataset
                print(f"Found {len(label_freqs)} labels with frequencies from Music4All dataset.")
            elif args.dataset == 'mmimdb':
                # For MMIMDB (using Clip/Blip/LLaVaDataset), calculate from item_genres
                if not hasattr(temp_train_dataset, 'item_genres'):
                    raise AttributeError("MMIMDB dataset class needs 'item_genres' attribute for frequency calculation.")
                all_genres_flat = [genre for sublist in temp_train_dataset.item_genres for genre in sublist]
                label_freqs = Counter(all_genres_flat)
                all_possible_labels = get_mmimdb_labels()  # Get the canonical list
                print(f"Calculated {len(label_freqs)} label frequencies from MMIMDB item genres.")
            else:
                raise NotImplementedError(f"Frequency calculation not implemented for dataset type associated with {type(temp_train_dataset)}")

            if not label_freqs:
                raise ValueError("Could not determine label frequencies.")

            # Sort labels by frequency (descending)
            sorted_labels = sorted(label_freqs.keys(), key=lambda x: label_freqs.get(x, 0), reverse=True)

            # Filter sorted_labels to only include those present in the canonical list (important for MMIMDB)
            # And handle case where requested N > available N
            valid_sorted_labels = [lbl for lbl in sorted_labels if lbl in all_possible_labels]
            num_to_select = min(args.svm_num_classes, len(valid_sorted_labels))
            selected_labels = valid_sorted_labels[:num_to_select]

            if not selected_labels:
                raise ValueError(f"Could not select any valid top-{args.svm_num_classes} labels.")
            print(f"Selected Top-{len(selected_labels)} Labels: {selected_labels}")
            num_labels = len(selected_labels)
            del temp_train_dataset

        else:
            # Default: Use all labels defined by the dataset
            print("\nUsing all available classes for the dataset.")
            if args.dataset == 'mmimdb':
                selected_labels = get_mmimdb_labels()
            elif args.dataset == 'music4all':
                # Need to instantiate dataset to get genres if not done above
                temp_ds = load_examples(args, tokenizer, evaluate=False, labels_override=None)
                selected_labels = temp_ds.genres
                del temp_ds
            else:
                raise ValueError("Cannot determine default labels for unknown dataset.")
            num_labels = len(selected_labels)
            print(f"Using {num_labels} default labels: {selected_labels[:5]}...")  # Print first few

        # Load Datasets with Determined Labels
        print(f"\nLoading train dataset using the {num_labels} selected labels...")
        train_dataset_full = load_examples(args, tokenizer, evaluate=False, labels_override=selected_labels)
        print(f"Full train dataset size (mapped to {num_labels} labels): {len(train_dataset_full)}")

        print(f"\nLoading eval dataset using the {num_labels} selected labels...")
        eval_dataset = load_examples(args, tokenizer, evaluate=True, labels_override=selected_labels)
        print(f"Full eval dataset size (mapped to {num_labels} labels): {len(eval_dataset)}")

        # Filter Training Data (items must have at least one of the selected labels)
        print("\nFiltering training dataset for items containing at least one target label...")
        valid_indices = [
            i for i, item in enumerate(train_dataset_full)
            if item['label'].sum() > 0  # Check if the label vector (mapped to selected N classes) is not all zeros
        ]
        if not valid_indices:
            raise ValueError("No items found with any of the selected target labels in the training set. Cannot proceed.")
        print(f"Found {len(valid_indices)} items with at least one target label.")
        filtered_train_dataset = Subset(train_dataset_full, valid_indices)

        if args.svm_num_samples is not None:
            print(f"\nAttempting to sample {args.svm_num_samples} training items...")
            num_available_for_sampling = len(filtered_train_dataset)

            if num_available_for_sampling < args.svm_num_samples:
                print(f"Warning: Only {num_available_for_sampling} valid items available after filtering, "
                      f"less than the requested {args.svm_num_samples}. Using all {num_available_for_sampling} items.")
                sampled_indices_local = list(range(num_available_for_sampling))  # Use all indices from the filtered set
                num_samples_used_for_training = num_available_for_sampling
            else:
                # Sample from the indices *of the filtered_train_dataset*
                sampled_indices_local = random.sample(range(num_available_for_sampling), args.svm_num_samples)
                num_samples_used_for_training = args.svm_num_samples
                print(f"Randomly selected {num_samples_used_for_training} indices for training.")

            # Create the final Subset dataset for training feature extraction
            sampled_train_dataset = Subset(filtered_train_dataset, sampled_indices_local)
            dataset_for_svm_train_features = sampled_train_dataset
            print(f"Final training subset size for feature extraction: {len(dataset_for_svm_train_features)}")
        else:
            # No sampling requested, use the full filtered dataset
            print("\nUsing all items that contain target labels for training (no sampling requested).")
            dataset_for_svm_train_features = filtered_train_dataset
            print(f"Training subset size for feature extraction: {len(dataset_for_svm_train_features)}")

        # Extract Features and Train SVM
        print(f"\nExtracting features from {len(dataset_for_svm_train_features)} training samples...")
        feature_extractor_train = SVMFeatureExtractor(dataset_for_svm_train_features, args, tokenizer)
        X_train, y_train = feature_extractor_train.extract_features()

        if y_train.shape[1] != num_labels:
            raise ValueError(f"Label dimension mismatch after extraction. Expected {num_labels}, got {y_train.shape[1]}.")

        print(f"\nInitializing SVMClassifier with {num_labels} labels...")
        svm_model = SVMClassifier(args, num_labels=num_labels)

        print(f"Training SVM with {X_train.shape[0]} samples, {X_train.shape[1]} features...")
        svm_model.fit(X_train, y_train)

        print(f"\nCalculating training metrics on the {X_train.shape[0]} samples used for fitting...")
        train_preds = svm_model.predict(X_train)
        train_f1 = f1_score(y_train, train_preds, average="samples", zero_division=0)
        train_prec = precision_score(y_train, train_preds, average="samples", zero_division=0)
        train_rec = recall_score(y_train, train_preds, average="samples", zero_division=0)
        print(f"Training Metrics: F1={train_f1:.4f}, P={train_prec:.4f}, R={train_rec:.4f}")
        if args.use_wandb:
            wandb.log({"train_f1_on_fit_data": train_f1, "train_precision_on_fit_data": train_prec, "train_recall_on_fit_data": train_rec, "num_train_samples_used": X_train.shape[0]})

        # Extract Evaluation Features and Evaluate
        print(f"\nExtracting features from the FULL evaluation set ({len(eval_dataset)} samples)...")
        feature_extractor_eval = SVMFeatureExtractor(eval_dataset, args, tokenizer)
        X_eval, y_eval = feature_extractor_eval.extract_features()

        if y_eval.shape[1] != num_labels:
            raise ValueError(f"Eval label dimension mismatch. Expected {num_labels}, got {y_eval.shape[1]}.")

        print(f"\nEvaluating SVM model on {X_eval.shape[0]} evaluation samples...")
        eval_preds = svm_model.predict(X_eval)

        result = {
            "Validation F1": f1_score(y_eval, eval_preds, average="samples", zero_division=0),
            "Validation Precision": precision_score(y_eval, eval_preds, average="samples", zero_division=0),
            "Validation Recall": recall_score(y_eval, eval_preds, average="samples", zero_division=0),
            "Validation Hamming Loss": hamming_loss(y_eval, eval_preds)
        }

        # Log and display results
        if args.use_wandb:
            wandb.log(result)
        print("SVM Evaluation Metrics:", result)

        return result
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.save_model:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        else:
            logger.info("Model saving disabled.")

    if args.do_eval and args.save_model:
        saved_model_path = os.path.join(args.output_dir, WEIGHTS_NAME)
        if os.path.exists(saved_model_path):
            model.load_state_dict(torch.load(saved_model_path))
            model.to(args.device)
        else:
            raise FileNotFoundError("Saved model weights not found at: {}".format(saved_model_path))

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        criterion = nn.BCEWithLogitsLoss()
        if args.save_model:
            checkpoints = [args.output_dir]
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                if args.model_type == "mmbt":
                    model = Custom_MMBTForClassification(config, transformer, img_encoder)
                elif args.model_type == "vilt":
                    model = Custom_ViLTForClassification(args, config, num_labels)

                model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
                model.to(args.device)
                result = evaluate(args, model, tokenizer, criterion, prefix=prefix)
                result = {k + "_{}".format(global_step): v for k, v in result.items()}
                results.update(result)
        else:
            logger.info("Evaluating using the in-memory model as per --no_save_model")
            result = evaluate(args, model, tokenizer, criterion)
            results.update(result)

    return results


if __name__ == "__main__":
    main()
    wandb.finish()
