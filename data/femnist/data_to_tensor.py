"""
Converts a list of (writer, [list of (file,class)]) into torch.tensor,
For each writer, creates a `.pt` file containing `data` and `targets`,
The resulting file is saved in `intermediate/data_as_tensor_by_writer'
"""
import os
import pickle

import torch
import numpy as np
from tqdm import tqdm

from PIL import Image


def relabel_class(c):
    """
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    """
    if c.isdigit() and int(c) < 40:
        return int(c) - 30
    elif int(c, 16) <= 90:  # uppercase
        return int(c, 16) - 55
    else:
        return int(c, 16) - 61


if __name__ == "__main__":
    by_writers_dir = os.path.join('intermediate', 'images_by_writer.pkl')
    save_dir = os.path.join('intermediate', 'data_as_tensor_by_writer')

    os.makedirs(save_dir, exist_ok=True)

    with open(by_writers_dir, 'rb') as f:
        writers = pickle.load(f)

    for (w, l) in tqdm(writers):

        data = []
        targets = []

        size = 28, 28  # original image size is 128, 128
        for (f, c) in l:
            file_path = os.path.join(f)
            img = Image.open(file_path)
            gray = img.convert('L')
            gray.thumbnail(size, Image.ANTIALIAS)
            arr = np.asarray(gray).copy() / 255  # scale all pixel values to between 0 and 1

            nc = relabel_class(c)

            data.append(arr)
            targets.append(nc)

        if len(data) > 2:
            data = torch.tensor(np.stack(data))
            targets = torch.tensor(np.stack(targets))

            trgt_path = os.path.join(save_dir, w)

            torch.save((data, targets), os.path.join(save_dir, f"{w}.pt"))
