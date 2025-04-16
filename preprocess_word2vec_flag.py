'''
This program checks which tags from the MyFonts dataset can be encoded by word2vec. 
It assigns a flag to each tag: 0 if it cannot be encoded, 1 if it can be directly encoded,  
2 if it can be encoded by replacing hyphens with underscores, and 3 if all hyphen-separated parts are encodable.  
The results are then saved to a CSV file.
'''

import gensim
import csv
import utils


WORD2VEC_PATH = 'GoogleNews-vectors-negative300.bin' # Path to the word2vec
SAVE_PATH = 'dataset/word2vec_flag.csv'


# Load the vocabulary from word2vec
word2vec = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
vocab = set(word2vec.index_to_key)

# Extract impression tags from the MyFonts dataset
tag_list = set([])
for dataset in ['train', 'val', 'test']:
    fontnames = utils.get_fontnames(dataset)
    for fontname in fontnames:
        tags = utils.get_org_tags(fontname)
        tags = [tag for tag in tags if tag not in utils.INVALID_TAGS]
        tag_list = tag_list | set(tags)
tag_list = sorted(tag_list)

# Set a flag indicating whether each tag can be encoded by word2vec
vocab_flag = {}
for tag in tag_list:
    # 0: the tag cannot be encoded
    flag = 0
    
    # 1: the tag itself can be directly encoded
    if tag in vocab:
        flag = 1    

    # 2: the tag with hyphens replaced by underscores can be encoded
    elif tag.replace('-', '_') in vocab:
        flag = 2
    
    # 3: all parts of the tag split by hyphens can be encoded
    elif all(split in vocab for split in tag.split('-')):
        flag = 3
    vocab_flag[tag] = flag

# Save the results
with open(SAVE_PATH, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for tag in tag_list:
        writer.writerow([tag, vocab_flag[tag]])