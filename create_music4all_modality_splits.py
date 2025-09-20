import os
import json
import random
from collections import OrderedDict


def load_split(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return set(data['train'])


def get_all_mbids(dataset_dir, data_type):
    all_mbids = set()
    data_dir = os.path.join(dataset_dir, data_type)
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            all_mbids.add(filename.split('.')[0])
    return all_mbids


def create_subsets(test_set, percentages):
    subsets = OrderedDict()
    sorted_test = sorted(test_set)
    random.seed(42)

    for p in sorted(percentages):
        if p == 100:
            subsets[p] = sorted_test
        else:
            subset_size = int(len(test_set) * (p / 100))
            if p == 10:
                subsets[p] = random.sample(sorted_test, subset_size)
            else:
                previous_subset = subsets[next(reversed(subsets))]
                additional_items = random.sample(list(set(sorted_test) - set(previous_subset)),
                                                 subset_size - len(previous_subset))
                subsets[p] = sorted(previous_subset + additional_items)

    return subsets


def create_modality_splits(dataset_dir, data_type, split_file, output_file):
    split_file = os.path.join(dataset_dir, split_file)
    output_file_text = os.path.join(dataset_dir, output_file)

    train_set = load_split(split_file)
    all_mbids = get_all_mbids(dataset_dir, data_type)
    test_set = all_mbids - train_set

    percentages = [10, 30, 50, 70, 90, 100]
    subsets = create_subsets(test_set, percentages)

    splits = {
        'train': sorted(train_set),
        'missing_modality': {str(p): subset for p, subset in subsets.items()}
    }

    # Save modality splits
    with open(output_file_text, 'w') as f:
        json.dump(splits, f, indent=2)



def main():
    dataset_dir = 'dataset'

    # Process albums
    create_modality_splits(
        dataset_dir,
        'album',
        'music4all_album_split.json',
        'album_modality_splits.json'
    )

    # Process artists
    create_modality_splits(
        dataset_dir,
        'artist',
        'music4all_artist_split.json',
        'artist_modality_splits.json',
    )


if __name__ == '__main__':
    main()