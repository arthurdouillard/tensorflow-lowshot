import os
import glob

import numpy as np
import numpy.random as rng
from PIL import Image


def parse_omniglot(omniglot_folder):
    def parse(folder, c):
        x, y = [], []
        mapping = {}

        for alphabet in os.listdir(folder):
            if alphabet[0] == '.': continue # Hidden file.

            for character in os.listdir(os.path.join(folder, alphabet)):
                if character[0] == '.': continue # Hidden file.

                # A class is a unique character from an alphabet.
                mapping[c] = f'{alphabet}_{character}'

                for img in glob.glob(os.path.join(folder, alphabet, character, '*.png')):
                    y.append(c)
                    x.append(img)

                c += 1

        return (np.array(x), np.array(y)), mapping, c

    train, mapping, c = parse(os.path.join(omniglot_folder, 'images_background'), 0)
    test, test_mapping, _ = parse(os.path.join(omniglot_folder, 'images_evaluation'), c)
    mapping.update(test_mapping)

    return train, test, mapping


def open_image(path, target_size, normalize=True):
    img = Image.open(path).resize(target_size).convert('RGB')
    arr = np.asarray(img)

    if len(arr.shape) == 2: # If single channel image, add a dummy dimmension.
        arr = np.expand_dims(arr, axis=-1)

    if normalize:
        arr = arr / 127.5 - 1

    return arr


class SiameseLoader:
    def __init__(self, x, y, batch_size=32, target_size=(105, 105), shuffle=True,
                 seed=1337):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.n_classes = len(np.unique(self.y))

        rng.seed(seed)

    def get_batch(self, batch_size=None):
        batch_size = batch_size or self.batch_size # Overriding default batch size.

        x = np.empty((2, batch_size, *self.target_size, 3))
        y = np.zeros((batch_size))

        idx = 0
        classes = rng.choice(
            self.n_classes,
            batch_size,
            replace=False
        )

        # Same classes
        for cls in classes[:batch_size // 2]:
            try:
                paths = rng.choice(self.x[np.where(self.y == cls)[0]], 2)
            except:
                print(cls)
                print(np.where(self.y == cls))
            x[0, idx] = open_image(paths[0], self.target_size)
            x[1, idx] = open_image(paths[1], self.target_size)
            idx += 1

        # Different classes
        for cls in classes[batch_size // 2: batch_size]:
            left = rng.choice(self.x[np.where(self.y == cls)[0]], 1)[0]
            right = rng.choice(self.x[np.where(self.y != cls)[0]], 1)[0]

            x[0, idx] = open_image(left, self.target_size)
            x[1, idx] = open_image(right, self.target_size)
            y[idx] = 1
            idx += 1

        if self.shuffle:
            idxes = rng.permutation(batch_size)
            x = x[:, idxes, :, :, :]
            y = y[idxes]

        return x[0], x[1], y
