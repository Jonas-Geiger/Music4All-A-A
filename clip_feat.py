import os
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image, ImageFile
from transformers import CLIPModel, AutoProcessor, AutoTokenizer
from tqdm import tqdm
import argparse
import sys 
import re

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_nested_value(data_dict, keys, default=None):
    """Safely retrieve a nested value from a dictionary."""
    value = data_dict
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError, IndexError):
        return default

def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features for MMIMDB or Music4All dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mmimdb", "music4all"],
                        help="Dataset to process (mmimdb or music4all).")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing the dataset folders (e.g., /opt/datasets).")
    parser.add_argument("--music4all_data_type", type=str, default=None, choices=["artist", "album"],
                        help="Type of data for Music4All (artist or album). Required if --dataset is music4all.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (default: 0)")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14",
                        help="CLIP model to use (default: openai/clip-vit-large-patch14)")
    parser.add_argument("--saving_batch_size", type=int, default=1000,
                        help="Batch size for saving results to CSV (default: 1000)")
    parser.add_argument("--max_text_length", type=int, default=77,
                        help="Maximum text length in tokens (default: 77, CLIP's limit)")
    args = parser.parse_args()

    # Input Validation
    if args.dataset == "music4all" and not args.music4all_data_type:
        parser.error("--music4all_data_type is required when --dataset is music4all")
    if args.dataset == "mmimdb" and args.music4all_data_type:
        print("[WARNING] --music4all_data_type is ignored when --dataset is mmimdb")
        args.music4all_data_type = None # Ensure it's None

    # Set Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[INFO] Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("[INFO] CUDA not available, using CPU.")

    # Configure Paths and Settings based on Dataset
    dataset_name = args.dataset
    data_subdir = "" # Subdirectory within base_dir for JSON/Image files
    split_filename = ""
    image_extension = ""
    output_prefix = f"clip_{dataset_name}" # Base prefix for output files
    text_extraction_keys = []
    base_output_dir = "" # Directory where output folders will be created

    if dataset_name == "mmimdb":
        data_subdir = os.path.join(dataset_name, "dataset")
        dataset_dir = os.path.join(args.base_dir, data_subdir) # Where JSON/Image files are
        split_file_path = os.path.join(args.base_dir, dataset_name, "split.json")
        image_extension = ".jpeg"
        text_extraction_keys = ["plot"] # Special handling for list below
        output_prefix = "clip" # Keep original naming for mmimdb for compatibility
        base_output_dir = os.path.join(args.base_dir, dataset_name) # Output within mmimdb folder

    elif dataset_name == "music4all":
        # Define the parent directory for splits and outputs
        music4all_base_subdir = os.path.join("mmimdb", "Music4All_new_modalities")
        music4all_base_path = os.path.join(args.base_dir, music4all_base_subdir)

        # Define the specific data directory (artist or album) for JSON/JPG files
        data_subdir = os.path.join(music4all_base_subdir, args.music4all_data_type)
        dataset_dir = os.path.join(args.base_dir, data_subdir)

        # Split file path is in the parent directory
        split_file_path = os.path.join(music4all_base_path, f"{args.music4all_data_type}_modality_splits.json")

        image_extension = ".jpg"
        output_prefix = f"clip_{args.music4all_data_type}"

        # Output directory is also relative to the parent Music4All directory
        base_output_dir = music4all_base_path

        if args.music4all_data_type == "artist":
            text_extraction_keys = ["artist_info", "artist", "wiki", "summary"]
        elif args.music4all_data_type == "album":
            text_extraction_keys = ["album_info", "album", "wiki", "summary"]

        # Add a check to ensure the data directory exists
        if not os.path.isdir(dataset_dir):
            print(f"[ERROR] The specific data directory for Music4All ({args.music4all_data_type}) was not found:")
            print(f"[ERROR] Looked for: {dataset_dir}")
            print(f"[ERROR] Please check your --base_dir and dataset structure.")
            sys.exit(1)

    else:
        print(f"[ERROR] Invalid dataset name: {dataset_name}")
        sys.exit(1)

    # Output Directories
    out_img_dir = os.path.join(base_output_dir, f"{output_prefix}_encoded_images")
    out_txt_dir = os.path.join(base_output_dir, f"{output_prefix}_encoded_texts")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_txt_dir, exist_ok=True)

    print(f"[INFO] Processing dataset: {dataset_name}" + (f" ({args.music4all_data_type})" if args.music4all_data_type else ""))
    print(f"[INFO] Base directory: {args.base_dir}")
    print(f"[INFO] Dataset directory: {dataset_dir}") # Where JSON/Images are read from
    print(f"[INFO] Loading split file from: {split_file_path}")
    print(f"[INFO] Image extension: {image_extension}")
    print(f"[INFO] Output image dir: {out_img_dir}")
    print(f"[INFO] Output text dir: {out_txt_dir}")

    # Load Split File
    try:
        with open(split_file_path, 'r') as f:
            splits_data = json.load(f)
        # Handle potential nested structure in music4all splits
        if dataset_name == "music4all" and "train" not in splits_data:
             if "train" in splits_data.get("modality_splits", {}):
                 splits = splits_data["modality_splits"]
                 print("[INFO] Found splits inside 'modality_splits' key.")
             elif "train" in splits_data.get("full_mbids", {}): # Another possible structure
                 # If only full lists are available, use them for both train/test
                 all_ids = splits_data["full_mbids"]["train"] # Assuming train has all
                 splits = {"train": all_ids, "test": all_ids}
                 print("[WARNING] Using 'full_mbids' for train/test splits as modality splits not found.")
             else:
                  raise KeyError("Could not find 'train'/'test' keys directly or nested in split file.")
        else:
            splits = splits_data

        print(f"[INFO] Found splits: {list(splits.keys())}")
    except Exception as e:
        print(f"[ERROR] Failed to load or parse split file '{split_file_path}': {e}")
        return

    # Load CLIP Model
    print(f"[INFO] Loading CLIP model: {args.model_name}")
    try:
        model = CLIPModel.from_pretrained(args.model_name).to(device)
        processor = AutoProcessor.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print("[INFO] CLIP model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load CLIP model: {e}")
        return

    # Feature Extraction Loop
    overall_total = 0
    overall_success = 0
    overall_fail = 0

    for split in ["train", "test"]:
        print(f"\n[INFO] Starting feature extraction for split: {split}")

        split_ids = []  # Initialize as empty
        if dataset_name == "music4all" and split == "test":
            if "missing_modality" in splits and isinstance(splits.get("missing_modality"), dict):
                # Use the "100" key for the full test
                test_key = "100"
                if test_key in splits["missing_modality"]:
                    split_ids = splits["missing_modality"].get(test_key, [])  # Use .get for safety
                    if split_ids:
                        print(f"[INFO] Found test split IDs under 'missing_modality' -> '{test_key}' key.")
                    else:
                        print(f"[WARNING] Key '{test_key}' under 'missing_modality' is empty. No test IDs.")
                else:
                    print(f"[WARNING] Test key '{test_key}' not found under 'missing_modality' in split file. Cannot process test split.")
            else:
                print("[WARNING] 'missing_modality' key not found or not a dictionary in split file. Cannot process test split.")
        else:
            # logic for 'train' split (both datasets) or 'test' split (mmimdb)
            split_ids = splits.get(split, [])
            if not split_ids and split == "test":
                print(f"[WARNING] Top-level 'test' key not found or empty for dataset '{dataset_name}'.")
            elif split_ids:
                print(f"[INFO] Found IDs for split '{split}' using top-level key.")

        if not split_ids:
            print(f"[WARNING] No IDs found for split '{split}'. Skipping.")
            continue

        total_samples = len(split_ids)
        print(f"[INFO] Total samples in '{split}': {total_samples}")

        img_feats = []
        txt_feats = []
        success_count = 0
        failure_count = 0
        failed_ids = []

        for item_id in tqdm(split_ids, desc=f"Processing {split} samples"):
            overall_total += 1
            json_path = os.path.join(dataset_dir, f"{item_id}.json")

            img_path = None
            img_filename = None

            try:
                # List files in the directory and find one starting with the ID and ending with the extension
                potential_img_files = [
                    f for f in os.listdir(dataset_dir)
                    if f.startswith(str(item_id)) and f.lower().endswith(image_extension.lower())
                ]

                if potential_img_files:
                    if len(potential_img_files) == 1:
                        # If only one image, use it directly
                        img_filename = potential_img_files[0]
                    else:
                        # Multiple images found, parse index and find the highest
                        highest_index = -1 # Reset for each item_id
                        best_filename = None
                        for fname in potential_img_files:
                            # Try to extract the index number using regex (looks for _N.ext at the end)
                            match = re.search(r'_(\d+)\.[^.]+$', fname)
                            if match:
                                current_index = int(match.group(1))
                                if current_index >= highest_index:
                                    highest_index = current_index
                                    best_filename = fname
                            # else: file doesn't match the _N.ext pattern, ignore for index comparison

                        # If we found a best file based on index, use it
                        if best_filename:
                            img_filename = best_filename
                        else:
                            # Fallback: If no files matched the pattern _N.ext,
                            # sort alphabetically and take the last one as a heuristic
                            potential_img_files.sort()
                            img_filename = potential_img_files[-1]
                            print(f"[DEBUG] Could not parse index for {item_id}, falling back to last file: {img_filename}")

                    # Construct the full path if a filename was selected
                    if img_filename:
                        img_path = os.path.join(dataset_dir, img_filename)

                # else: No potential files found, img_path remains None

            except FileNotFoundError:
                 print(f"[ERROR] Dataset directory not found when searching for image: {dataset_dir}")
                 failure_count += 1
                 overall_fail += 1
                 failed_ids.append(item_id)
                 continue # Skip to next item_id
            except Exception as e:
                 print(f"[ERROR] Error finding image file for {item_id} in {dataset_dir}: {e}")
                 failure_count += 1
                 overall_fail += 1
                 failed_ids.append(item_id)
                 continue # Skip to next item_id


            # File Existence Check (using potentially found img_path)
            if not os.path.isfile(json_path):
                print(f"[WARNING] JSON file not found for id: {item_id} at {json_path}")
                failure_count += 1
                overall_fail += 1
                failed_ids.append(item_id)
                continue
            # Check if an image path was successfully found and if that file exists
            if not img_path or not os.path.isfile(img_path):
                print(f"[WARNING] Image file not found for id: {item_id} (looked for pattern starting with ID)")
                failure_count += 1
                overall_fail += 1
                failed_ids.append(item_id)
                continue

            # Load JSON Data
            try:
                with open(json_path, 'r') as jf:
                    item_data = json.load(jf)
            except Exception as e:
                print(f"[ERROR] Failed to load JSON for id {item_id}: {e}")
                failure_count += 1
                overall_fail += 1
                failed_ids.append(item_id)
                continue

            # Extract Text
            text = ""
            if dataset_name == "mmimdb":
                plot_list = get_nested_value(item_data, text_extraction_keys, default=[])
                if isinstance(plot_list, list):
                    for plot_entry in plot_list:
                        if plot_entry and plot_entry.strip():
                            text = plot_entry.strip()
                            break
            elif dataset_name == "music4all":
                text = get_nested_value(item_data, text_extraction_keys, default="")
                if not isinstance(text, str): # Ensure it's a string
                    text = ""

            if not text or not text.strip():
                 # Use placeholder if text is empty or whitespace only
                 if dataset_name == "mmimdb":
                     text = "No plot available for this movie."
                 else: # music4all
                     text = f"No summary available for this {args.music4all_data_type}."

            # Load Image (using the found img_path)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Failed to open image {img_path} for id {item_id}: {e}")
                failure_count += 1
                overall_fail += 1
                failed_ids.append(item_id)
                continue

            # Extract CLIP Features
            try:
                # Extract image features
                with torch.no_grad():
                    img_inputs = processor(images=img, return_tensors="pt").to(device)
                    img_features = model.get_image_features(**img_inputs)
                    feat_img = img_features.detach().cpu().numpy()[0]
                    del img_inputs, img_features
                    torch.cuda.empty_cache()

                # Extract text features
                with torch.no_grad():
                    text_inputs = tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=args.max_text_length,
                        return_tensors="pt"
                    ).to(device)
                    text_features = model.get_text_features(**text_inputs)
                    feat_txt = text_features.detach().cpu().numpy()[0]
                    del text_inputs, text_features
                    torch.cuda.empty_cache()

                # Append features with ID
                img_feats.append([str(item_id)] + feat_img.tolist()) # Ensure ID is string
                txt_feats.append([str(item_id)] + feat_txt.tolist()) # Ensure ID is string
                success_count += 1
                overall_success += 1

            except Exception as e:
                print(f"[ERROR] CLIP feature extraction failed for id {item_id}: {e}")
                failure_count += 1
                overall_fail += 1
                failed_ids.append(item_id)
                continue
            finally:
                img.close()


        # Print Split Summary
        print(f"[INFO] Finished processing split '{split}'.")
        if total_samples > 0:
            success_percent = (success_count / total_samples) * 100
            fail_percent = (failure_count / total_samples) * 100
            print(f"[INFO] Split '{split}': Total: {total_samples}, Success: {success_count} ({success_percent:.2f}%), Failures: {failure_count} ({fail_percent:.2f}%).")
            print("List of Failures: {}".format(failed_ids))
        else:
             print(f"[INFO] Split '{split}': No samples processed.")

        if failure_count > 0:
             print(f"[INFO] Processed {success_count} items successfully, failed {failure_count} items.")


        # Save Features to CSV
        if img_feats:
            num_img_features = len(img_feats[0]) - 1
            img_header = ["item_id"] + [str(i) for i in range(num_img_features)]
            out_img_file = os.path.join(out_img_dir, f"{output_prefix}_images_latent_{split}.csv")

            df_img = pd.DataFrame(img_feats, columns=img_header)
            df_img.to_csv(out_img_file, index=False, chunksize=args.saving_batch_size)
            print(f"[INFO] Saved image features to: {out_img_file} (Shape: {df_img.shape})")
        else:
            print(f"[WARNING] No image features extracted for split '{split}'.")

        if txt_feats:
            num_txt_features = len(txt_feats[0]) - 1
            txt_header = ["item_id"] + [str(i) for i in range(num_txt_features)]
            out_txt_file = os.path.join(out_txt_dir, f"{output_prefix}_txt_latent_{split}.csv")

            df_txt = pd.DataFrame(txt_feats, columns=txt_header)
            df_txt.to_csv(out_txt_file, index=False, chunksize=args.saving_batch_size)
            print(f"[INFO] Saved text features to: {out_txt_file} (Shape: {df_txt.shape})")
        else:
            print(f"[WARNING] No text features extracted for split '{split}'.")

    print("\n[INFO] Feature extraction complete.")
    if overall_total > 0:
        overall_success_p = (overall_success / overall_total) * 100
        overall_fail_p = (overall_fail / overall_total) * 100
        print(f"[INFO] Overall samples processed: {overall_total}, Successes: {overall_success} ({overall_success_p:.2f}%), Failures: {overall_fail} ({overall_fail_p:.2f}%).")
    else:
        print("[INFO] No samples processed overall.")


if __name__ == "__main__":
    main()
