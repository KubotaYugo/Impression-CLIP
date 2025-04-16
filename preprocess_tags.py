'''
This program processes impression tags from the MyFonts dataset and generates a refined tag set for each font.
It first excludes tags that are garbled or cannot be encoded by Word2Vec, then counts tag frequencies per dataset.
For each font, it selects the top 10 tags based on their frequency in the 'train' set.
Tags with a frequency lower than 50 in the 'train' set are removed.
The filtered tag lists and tag frequency statistics are saved to CSV files.
'''


import os
import csv
from collections import defaultdict
import utils


WORD2VEC_FLAG_PATH = 'dataset/word2vec_flag.csv'
SAVE_PATH = 'dataset/tags'


# Get the font names for train, val, and test datasets
fontnames = {
    'train':utils.get_fontnames('train'), 
    'val':utils.get_fontnames('val'), 
    'test':utils.get_fontnames('test')
    }

os.makedirs(SAVE_PATH, exist_ok=True)


# ==================================================
# (1) Exclude garbled tags and those that cannot be encoded by word2vec, then count original tag frequencies
# ==================================================
# Load flags indicating whether tags can be encoded with word2vec
word2vec_flag = dict(utils.load_csv(WORD2VEC_FLAG_PATH))

# Get tag frequencies (excluding invalid and unencodable tags)
tag_freq_org = {ds: {} for ds in ['train', 'val', 'test']}
for dataset in ['train', 'val', 'test']:
    for fontname in fontnames[dataset]:
        for tag in utils.get_org_tags(fontname):
            if (tag not in utils.INVALID_TAGS) and (word2vec_flag.get(tag, '0') != '0'):
                tag_freq_org[dataset][tag] = tag_freq_org[dataset].get(tag, 0) + 1

# Collect all unique tags across datasets
tag_list = sorted(set().union(*[tag_freq_org[ds].keys() for ds in tag_freq_org]))

# Ensure all tags exist in each dataset's frequency dict (with 0 if missing)
for tag in tag_list:
    for dataset in tag_freq_org:
        tag_freq_org[dataset].setdefault(tag, 0)


# ==================================================
# (2) For each font, sort its tags by frequency in 'train' and keep the top 10 tags
# ==================================================
tags_dict_top10 = {ds: {} for ds in ['train', 'val', 'test']}
for dataset in ['train', 'val', 'test']:
    for fontname in fontnames[dataset]:
        tags = [tag for tag in utils.get_org_tags(fontname) if tag in tag_freq_org['train']]
        # Sort tags by descending frequency in the 'train' set, and keep top 10
        tags_sorted = sorted(tags, key=lambda x: -tag_freq_org['train'].get(x, 0))
        tags_dict_top10[dataset][fontname] = tags_sorted[:10]

# Compute tag frequencies after keeping only top-10 tags per font
tag_freq_top10 = {ds: defaultdict(int) for ds in ['train', 'val', 'test']}
for ds in ['train', 'val', 'test']:
    for tags in tags_dict_top10[ds].values():
        for tag in tags:
            tag_freq_top10[ds][tag] += 1


# ==================================================
# (3) Sort tags by top-10 frequency in the 'train' set
# ==================================================
tag_list_sorted_top10 = sorted(tag_list, key=lambda tag: tag_freq_top10['train'].get(tag, 0), reverse=True)

# Sort each font's tag list based on global top-10 order
tags_dict_top10_sorted = {
    ds: {
        fontname: sorted(tags, key=tag_list_sorted_top10.index)
        for fontname, tags in tags_dict_top10[ds].items()
    }
    for ds in ['train', 'val', 'test']
}

# Save the sorted tag frequency table
with open(f'{SAVE_PATH}/tag_freq.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['tag', 'total', 'train', 'val', 'test'])
    for tag in tag_list_sorted_top10:
        row = [tag] + [sum(tag_freq_top10[ds].get(tag, 0) for ds in ['train', 'val', 'test'])] + \
              [tag_freq_top10[ds].get(tag, 0) for ds in ['train', 'val', 'test']]
        writer.writerow(row)


# ==================================================
# (4) After narrowing to top-10, remove tags with <50 frequency in 'train' and save the remaining tags
# ==================================================
# Create necessary directories
for subset in ['train', 'val', 'test']:
    os.makedirs(os.path.join(SAVE_PATH, subset), exist_ok=True)

# Process and save filtered tags per font
for dataset in ['train', 'val', 'test']:
    for fontname, tags in tags_dict_top10_sorted[dataset].items():
        # Filter out tags with train frequency < 50
        filtered_tags = [tag for tag in tags if tag_freq_top10['train'].get(tag, 0) >= 50]
        if filtered_tags == []:
            print(fontname)
            continue  # Skip if no valid tags remain
        # Save tags to CSV
        save_file = os.path.join(SAVE_PATH, dataset, f'{fontname}.csv')
        with open(save_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(filtered_tags)