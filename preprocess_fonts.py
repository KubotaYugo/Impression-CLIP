'''
Preprocessing fonts from the MyFonts dataset.
For each font, extract uppercase characters, find the maximum height or width among them,
normalize the size of each character using MAX_LENGTH / that maximum value, concatenate the images, and save the result.
'''

from PIL import Image
import numpy as np
import string
import os
from decimal import Decimal, ROUND_HALF_UP
from multiprocessing import pool
import utils


MAX_LENGTH = 64     # Side length of the output image
SAVE_PATH = 'dataset/fonts'
UPPER_CASE = string.ascii_uppercase


def round_to_first_decimal(value):
    rounded_value = Decimal(str(value)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    return int(rounded_value)

def calc_corner(img):
    y_min = np.where(img!=255)[0].min()
    y_max = np.where(img!=255)[0].max()
    x_min = np.where(img!=255)[1].min()
    x_max = np.where(img!=255)[1].max()
    return y_min, y_max, x_min, x_max

def save_fonts(values):
    i = values[0]
    dataset = values[1]
    fontname = values[2]
    print(dataset, i)

    # Get paths for all characters in font_name
    char_paths = [f'{utils.MYFONTS_PATH}/fontimage/{fontname}_{char*2}.png' for char in UPPER_CASE]

    # Get the largest height or width among all characters in the font
    max_length_font = 0
    for char_path in char_paths: # Consider only uppercase characters
        img = np.array(Image.open(char_path).convert('L')) 
        if len(np.where(img!=255)[0])!=0:
            y_min, y_max, x_min, x_max = calc_corner(img)
            max_length_char = np.max([y_max-y_min, x_max-x_min])
            if max_length_char > max_length_font:
                max_length_font = max_length_char

    # Scaling factor to resize characters in the font to fit within MAX_LENGTH
    rate = MAX_LENGTH / max_length_font 
    
    img_list = []
    for char_path in char_paths:
        img = np.array(Image.open(char_path).convert('L')) 

        # Crop the character region
        y_min, y_max, x_min, x_max = calc_corner(img)
        img_clip = img[y_min:y_max+1, x_min:x_max+1]
        img_clip = Image.fromarray(img_clip.astype(np.uint8))

        # Resize the character region
        width = round_to_first_decimal(rate*(x_max-x_min))
        height = round_to_first_decimal(rate*(y_max-y_min))
        width = 1 if width == 0 else width
        height = 1 if height == 0 else height
        img_resize = img_clip.resize([width, height], resample=Image.BICUBIC)
        img_resize = np.array(img_resize)

        # Paste the character onto the center of a white square image
        img_square = np.full([MAX_LENGTH, MAX_LENGTH], 255)
        height = img_resize.shape[0]
        width = img_resize.shape[1]
        y_top = round_to_first_decimal((MAX_LENGTH-height)/2)
        x_left = round_to_first_decimal((MAX_LENGTH-width)/2)
        img_square[y_top:y_top+height, x_left:x_left+width] = img_resize

        # Normalize pixel values to the range 0–1
        img_square = img_square - np.min(img_square)
        img_square = img_square / np.max(img_square)
        img_list.append(img_square)

    # Save as a compressed numpy file
    np.savez_compressed(f'{SAVE_PATH}/{dataset}/{fontname}.npz', np.array(img_list))

if __name__ == '__main__':
    for dataset in ['test', 'val', 'train']:
        fontnames = utils.get_fontnames(dataset)
        os.makedirs(f'{SAVE_PATH}/{dataset}', exist_ok=True)

        values = [[i, dataset, fontnames[i]] for i in range(len(fontnames))]
        
        p = pool.Pool(5)
        p.map(save_fonts, values)

        # for value in values:
        #     save_fonts(value)