import torch
import os
import numpy as np
import json
from torch.utils.data import Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import set_seed
import random
import math
import sys
from loguru import logger
import mmap

def log_rank_0(msg, rank=0):
    if rank <= 0:
        logger.info(msg)

def reset_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class BaseDataset(Dataset):
    def __init__(self, data_mmaps, data_pos_mmaps, selected_indices, tokenizer, max_seq_len, converter):
        self.data_mmaps = data_mmaps
        self.data_pos_mmaps = data_pos_mmaps
        self.selected_indices = selected_indices
        self.data_index_list = [0]
        for i, indices_set in enumerate(self.selected_indices):
            self.data_index_list.append(self.data_index_list[i] + len(indices_set))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.converter = converter
        self.last_data = None
    
    def __len__(self):
        return self.data_index_list[-1]

    def __getitem__(self, idx):
        outside_index = 0 # 当前要拿到的数据属于哪一个data_path
        inside_index = 0 # 当前要拿到的数据属于这个data_path的哪一个数据
        for i, index in enumerate(self.data_index_list):
            if idx < index:
                outside_index = i - 1
                inside_index = idx - self.data_index_list[i - 1]
                break
        selected_index = self.selected_indices[outside_index][inside_index]
        # 通过mmap机制，快速读取到数据
        sample_begin_index = int(self.data_pos_mmaps[outside_index][selected_index * 16: (selected_index + 1) * 16])
        sample_end_index = int(self.data_pos_mmaps[outside_index][(selected_index + 1) * 16: (selected_index + 2) * 16])
        data_str = self.data_mmaps[outside_index][sample_begin_index:sample_end_index]
        # 解析出来数据
        try:
            data = json.loads(data_str)
        except:
            data = self.last_data
        self.last_data = data
        # 进行数据转换
        return self.converter(data, self.tokenizer, self.max_seq_len)

class DatasetSpliter:
    def __init__(self, data_config, local_rank, rank, world_size, seed, tokenizer, converter, max_seq_len=None):
        if isinstance(data_config, str):
            with open(data_config, 'r') as f:
                data_config = json.load(f)
        data_paths = data_config['data_paths']
        data_output_dir = data_config['data_output_dir']
        train_proportion = data_config['train_proportion']
        eval_proportion = data_config['eval_proportion']
        data_pos_paths = []
        for data_path in data_paths:
            filename = data_path.replace('/', '_').split('.')[0] + '_pos.txt'
            data_pos_path = os.path.join(data_output_dir, filename)
            data_pos_paths.append(data_pos_path)
        if rank <= 0:
            self.preprocess(data_paths, data_pos_paths, data_output_dir)
        torch.distributed.barrier()
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        #! 有指定的话就优先
        if not max_seq_len:
            max_seq_len = data_config['max_seq_len']
        # else:
        #     max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id
        reset_random_seed(seed)
        self.split_and_random_dataset(data_paths, data_pos_paths, train_proportion, eval_proportion)
        self.train_dataset_ = BaseDataset(self.data_mmaps, self.data_pos_mmaps, self.selected_train_indices, tokenizer, max_seq_len, converter)
        self.eval_dataset_ = BaseDataset(self.data_mmaps, self.data_pos_mmaps, self.selected_eval_indices, tokenizer, max_seq_len, converter)

    # 预处理数据，计算每个data_path文件中，每行json sample的字节长度，用于后续mmap快速读取每个sample
    def preprocess(self, data_paths, data_pos_paths, data_output_dir):
        os.makedirs(data_output_dir, exist_ok=True)
        for data_path, data_pos_path in zip(data_paths, data_pos_paths):
            with open(data_path, 'r') as in_f:
                with open(data_pos_path, 'w') as out_f:
                    out_f.write('0000000000000000')
                    sum_size = 0
                    for line in in_f:
                        # 计算每一行的json字符串在文件的终止位置，并且将这个位置填充到16位，写入文件。
                        line_size = len(line.encode())
                        sum_size += line_size
                        out_f.write(str(sum_size).zfill(16))

    @property
    def train_dataset(self):
        return self.train_dataset_

    @property
    def eval_dataset(self):
        return self.eval_dataset_

    def split_and_random_dataset(self, data_paths, data_pos_paths, train_proportion, eval_proportion):
        self.data_mmaps = []
        self.data_pos_mmaps = []
        self.selected_train_indices = []
        self.selected_eval_indices = []
        bytes_per_pos = 16
        for i, data_path in enumerate(data_paths):
            with open(data_path, 'r') as f:
                m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.data_mmaps.append(m)
            with open(data_pos_paths[i], 'r') as f:
                m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.data_pos_mmaps.append(m)
                data_size = int(len(m) / bytes_per_pos) - 1
            # 将所有数据的indices进行打乱
            random_indices = list(np.random.permutation(data_size))
            # 将打乱的indices进行切分，切分出来训练集和测试集的indices
            train_indices, eval_indices = self.split_indices(random_indices, train_proportion, eval_proportion)
            selected_train_indices = self.selected_indices(train_indices)
            selected_eval_indices = self.selected_indices(eval_indices)
            self.selected_train_indices.append(selected_train_indices)
            self.selected_eval_indices.append(selected_eval_indices)

    def split_indices(self, random_indices, train_proportion, eval_proportion):
        data_size = len(random_indices)
        splits = [float(train_proportion), float(eval_proportion)]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size
        return random_indices[splits_index[0]:splits_index[1]], random_indices[splits_index[1]:]

    # 每个GPU进程，根据rank，拿到自己要处理的那部分indices
    def selected_indices(self, random_indices):
        data_size = len(random_indices)
        num_sample = int(math.ceil(data_size * 1.0 / self.world_size))
        begin_index = num_sample * self.rank
        end_index = num_sample * (self.rank + 1)
        if end_index <= data_size:
            selected_indices = random_indices[begin_index:end_index]
        elif begin_index < data_size:
            begin_index -= (end_index - data_size)
            end_index = data_size
            selected_indices = random_indices[begin_index:end_index]
        else:
            begin_index -= data_size
            end_index -= data_size
            selected_indices = random_indices[begin_index:end_index]
        return selected_indices

class LocalSampler(DistributedSampler):
    def __init__(self, dataset, rank, seed, start_epoch=0, start_iter=0, insert_after_index=0):
        self.dataset = dataset
        self.rank = rank
        assert start_iter >= insert_after_index, f'start_iter should larger than insert_after_index, {start_iter} vs {insert_after_index}'
        self.epoch = start_epoch
        self.start_iter = start_iter
        self.num_samples = len(self.dataset)
        self.seed = seed
        self.init = False
        self.insert_after_index = insert_after_index

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = list(range(len(self.dataset)))
        # 从插入的新数据开始
        indices = indices[self.insert_after_index:]
        random.seed(self.epoch + self.rank)
        # 根据epoch决定seed，并将indices打乱。这样可以避免每个epoch，处理的数据顺序都一样，这样模型性能会较差
        random.shuffle(indices)
        reset_random_seed(self.seed)
        # 从断点的数据开始，不用再次跑之前的数据
        indices = indices[self.start_iter - self.insert_after_index:]
        self.num_samples = len(indices)
        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.init:
            self.start_iter = 0
            self.insert_after_index = 0
        self.init = True

    def __len__(self):
        return self.num_samples