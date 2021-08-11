"""
Process SHAKESPEARE dataset, and splits it among clients
"""
import os
import re
import time
import random
import argparse

RAW_DATA_PATH = "raw_data/by_play_and_character"
TARGET_PATH = "all_data/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--s_frac',
        help='fraction of data to be used; default is 1.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--val_frac',
        help='fraction in validation set (from train set); default: 0.0;',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--train_tasks_frac',
        help='fraction of tasks / clients  participating to the training; default is 1.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes; default is 12345',
        type=int,
        default=12345,
        required=False)

    return parser.parse_args()


def train_test_split(raw_text, frac):
    r"""
    splits role text data into a set of training lines (the first `frac` of lines for the role),
     and test lines (the last 1 - `frac`, rounded up to at least one line)
    :param raw_text: raw text data
    :type raw_text: str
    :param frac: training fraction
    return `train_text`, `test_text`
    """
    assert 0 < frac < 1, "`frac` should be in (0, 1)"

    all_lines = raw_text.split('\n')[:-1]
    n_lines = len(all_lines)

    n_test_lines = max(1, int((1-frac)*n_lines))
    n_train_lines = n_lines - n_test_lines

    train_lines = all_lines[:n_train_lines]
    test_lines = all_lines[n_train_lines:]

    train_text = '\n'.join(train_lines)
    test_text = '\n'.join(test_lines)

    return train_text, test_text


def save_task(dir_path, train_text, test_text, val_text=None):
    r"""
    save `train_text` and `test_text` as `.txt` files in `dir_path`
    :param train_text:
    :param test_text:
    :param val_text:
    :param dir_path:
    """
    with open(os.path.join(dir_path, "train.txt"), 'w') as f:
        f.write(train_text)

    with open(os.path.join(dir_path, "test.txt"), 'w') as f:
        f.write(test_text)

    if val_text is not None:
        with open(os.path.join(dir_path, "val.txt"), 'w') as f:
            f.write(val_text)


def main():
    args = parse_args()

    rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)

    n_tasks = int(len(os.listdir(RAW_DATA_PATH)) * args.s_frac)
    file_names_list = os.listdir(RAW_DATA_PATH)
    rng.shuffle(file_names_list)

    file_names_list = file_names_list[:n_tasks]
    rng.shuffle(file_names_list)

    os.makedirs(os.path.join(TARGET_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_PATH, "test"), exist_ok=True)

    for idx, file_name in enumerate(file_names_list):
        if idx < int(args.train_tasks_frac * n_tasks):
            mode = "train"
        else:
            mode = "test"

        text_path = os.path.join(RAW_DATA_PATH, file_name)

        with open(text_path, "r") as f:
            raw_text = f.read()

        raw_text = re.sub(r"   *", r' ', raw_text)

        train_text, test_text = train_test_split(raw_text, args.tr_frac)

        if args.val_frac > 0:
            train_text, val_text = train_test_split(train_text, 1.-args.val_frac)
            val_text = val_text.replace('\n', ' ')

        else:
            val_text = None

        train_text = train_text.replace('\n', ' ')
        test_text = test_text.replace('\n', ' ')

        save_path = os.path.join(TARGET_PATH, mode, f"task_{idx}")
        os.makedirs(save_path, exist_ok=True)

        save_task(
            dir_path=save_path,
            train_text=train_text,
            test_text=test_text,
            val_text=val_text
        )


if __name__ == "__main__":
    main()
