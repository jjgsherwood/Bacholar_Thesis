import numpy as np
import os
import errno
import torch.utils.data as data

FILE_DICT = {
    "raw": ["Alina_art_1_1",
            "Alina_art_2_1",
            "Alina_art_4_2"],
    "smooth": ["Alina_art_1_1_sm",
               "Alina_art_2_1_sm",
               "Alina_art_4_2_sm"],
    "fluorescence": ["Alina_art_1_1_FL",
                     "Alina_art_2_1_FL",
                     "Alina_art_4_2_FL"],
    "raman": ["Alina_art_1_1_R",
              "Alina_art_2_1_R",
              "Alina_art_4_2_R"]
}

def check_for_files(path, needed_dict):
    file_dict = {}
    file_set = {k:set(f + '.npy' for f in v) for k,v in needed_dict.items()}
    for root, dirs, new_files in os.walk(path):
        for t, files in file_set.items():
            if files.issubset(set(new_files)):
                file_dict[t] = [os.path.join(root, f+'.npy') for f in needed_dict[t]]
    return file_dict

def search_for_folder(needed_dict, depth=3):
    file_dict = {}
    cur_dir = os.getcwd()
    while cur_dir != (cur_dir := os.path.dirname(cur_dir)):
        num_sep = cur_dir.count(os.path.sep)
        for root, dirs, _ in os.walk(cur_dir):
            if root.count(os.path.sep) >= num_sep + depth:
                break
            for name in dirs:
                if name.lower() == 'data':
                    file_dict.update(check_for_files(os.path.join(root, name), needed_dict))
                if file_dict.keys() == needed_dict.keys():
                    return file_dict

    return file_dict

def load_data(filenames):
    data = [np.load(f, 'r') for f in filenames]
    if len(data[0].shape) == 3:
        return np.concatenate([d.reshape(-1, d.shape[2]) for d in data], axis=0), [d.shape for d in data]
    return np.concatenate(data, axis=0)

def unit_vector_norm(X, *args):
    s = ((X**2).sum(axis=1))**0.5
    return (X.T / s).T, *[(x.T/s).T for x in args]

class RamanDataset(data.Dataset):
    def __init__(self, seq_length, mode="smooth", file_dir=None):
        self._seq_length = seq_length
        if mode == 'smooth':
            needed_keys = ('raw', 'smooth')
        elif mode == 'split':
            needed_keys = ('raw', 'raman', 'fluorescence')
        elif mode == 'all':
            needed_keys = ('raw', 'raman', 'fluorescence', 'smooth')
        else:
            raise ValueError("This mode is not known to the Dataset")

        if file_dir is None:
            needed_dict = {k:v for k,v in FILE_DICT.items() if k in needed_keys}
            file_dict = search_for_folder(needed_dict)
        else:
            file_dict = {key: [os.path.join(file_dir, v+'.npy') for v in value]
                         for key, value in FILE_DIR.items()}

        if file_dict.keys() == needed_keys:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                f'missing files for {" and ".join(list(needed_keys - file_dict.keys()))}')

        # data_dict = { for k,v in file_dict.items()}
        self._data, self._shape = load_data(file_dict['raw'])
        self._split = [s[0] * s[1] for s in self._shape]
        self._target = list((k,load_data(f)) for k,f in file_dict.items() if k != 'raw')
        self._target.sort()
        _, self._target = zip(*self._target)
        self._data, *self._target = unit_vector_norm(self._data, *self._target)

    @property
    def shape(self):
        return self._shape

    @property
    def sequence_len(self):
        return self._shape[0][2]

    def __getitem__(self, index):
        # check the index is not on the split
        return self._data[index], np.concatenate([x[index] for x in self._target])

    def __len__(self):
        return len(self._data)
