from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pickle
import numpy as np
import os
import random
import tqdm
import retriever.utils.utils as utils

from diskcache import FanoutCache


def getOxfordDataModule(cfg):
    return OxfordDataModule(cfg, data_class=eval(cfg['dataset_loader']))


cache = FanoutCache(directory=utils.CONFIG_DIR+"../data/cache",
                    shards=64,
                    timeout=1,
                    size_limit=3e11)

#################################################
#################### Oxford #####################
#################################################


def load_pc_file(filename, path):
    # returns Nx3 matrix
    pc = np.fromfile(os.path.join(path, filename), dtype=np.float64)

    if(pc.shape[0] != 4096*3):
        print("Error in pointcloud shape")
        return np.array([])

    pc = np.reshape(pc, (1, pc.shape[0]//3, 3))
    return pc.astype('float32')


def load_pc_files(filenames, path):
    pcs = []
    for filename in filenames:
        pc = load_pc_file(filename, path)
        pcs.append(pc)
    pcs = np.vstack(pcs)
    return pcs


@cache.memoize(typed=True)
def dict2bool(query_dict):
    with open(query_dict, 'rb') as handle:
        query_dict = pickle.load(handle)
    query_keys = list(query_dict.keys())
    n = len(query_keys)

    files = []
    is_pos = np.zeros([n, n], dtype=bool)
    is_neg = np.zeros([n, n], dtype=bool)
    for k in tqdm.tqdm(query_keys):
        files.append(query_dict[k]['query'])
        pos_idx = np.array(query_dict[k]['positives'], dtype=int)
        is_pos[k, pos_idx] = True
        neg_idx = np.array(query_dict[k]['negatives'], dtype=int)
        is_neg[k, neg_idx] = True
    return query_keys, files, is_pos, is_neg


class Oxford(Dataset):
    def __init__(self, query_dict, data_dir, num_pos=2, num_neg=18, return_only_query=False, validate=False):
        super(Oxford, self).__init__()

        # self.query_dict = None

        self.query_keys, self.files, self.is_pos, self.is_neg = dict2bool(
            query_dict)
        self.data_dir = data_dir
        self.num_pos = num_pos
        self.num_neg = num_neg

        self.return_only_query = return_only_query
        self.validate = validate
        print(f'Init Dataset done: {len(self.query_keys)}')

    def __len__(self):
        return len(self.query_keys)

    def load_pc_files(self, filenames, path):
        return load_pc_files(filenames, path)

    def load_pc_file(self, filenames, path):
        return load_pc_file(filenames, path)

    def __getitem__(self, index):
        query = self.load_pc_file(self.files[index], self.data_dir)
        if self.return_only_query:
            return {'query': query, 'query_idx': index}

        pos = np.where(self.is_pos[index, :])[0]
        np.random.shuffle(pos)
        pos_files = []

        act_num_pos = len(pos)
        if act_num_pos == 0:
            return self.__getitem__(index+1)
        for i in range(self.num_pos):
            pos_files.append(
                self.files[pos[i % act_num_pos]])
        #positives= load_pc_files(dict_value["positives"][0:num_pos])
        positives = self.load_pc_files(pos_files, self.data_dir)

        neg_files = []
        neg_indices = []
        hard_neg = []  # FIXME: have no hard neg
        if(len(hard_neg) == 0):
            neg = np.where(self.is_neg[index, :])[0]
            np.random.shuffle(neg)
            # random.shuffle(dict_value["negatives"])
            for i in range(self.num_neg):
                neg_files.append(
                    self.files[neg[i]])
                neg_indices.append(neg[i])

        negatives = self.load_pc_files(neg_files, self.data_dir)

        neighbors = []
        for pos_i in pos:
            neighbors.append(pos_i)
        for neg_i in neg_indices:
            for pos_i in np.where(self.is_pos[neg_i, :])[0]:
                neighbors.append(pos_i)
        possible_negs = list(set(self.query_keys)-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]

        neg2 = self.load_pc_file(
            self.files[possible_negs[0]], self.data_dir)
        # print(negatives)
        return {'query': query,
                'positives': positives,
                'negatives': negatives,
                'neg2': neg2,
                'is_pos': self.is_pos[index, :],
                'query_idx': index}
        # return [query, positives, negatives, neg2]


def loadNumpy(file, path=''):
    return np.load(os.path.join(path, file+'.npy'))[np.newaxis, ...].astype('float32')


class OxfordEmbedding(Oxford):
    def load_pc_file(self, filenames, path):
        return loadNumpy(filenames[:-4], path)
        # return torch.load(os.path.join(path, filenames+'.npy')).unsqueeze(0)

    def load_pc_files(self, filenames, path):
        return [loadNumpy(f[:-4], path)[0, ...] for f in filenames]
        # return [torch.load(os.path.join(path, f+'.npy')) for f in filenames]


def pad(array, n_points=2000):
    """ array [n x m] -> [n_points x m]
    """
    if len(array.shape) == 2:
        out = np.zeros([n_points, array.shape[-1]], dtype='float32')
        l = min(n_points, array.shape[-2])
        out[:l, :] = array[:l, :]
        return out
    else:
        size = list(array.shape)
        size[-2] = n_points
        out = np.zeros(size, dtype='float32')
        l = min(n_points, array.shape[-2])
        out[..., :l, :] = array[..., :l, :]
        # print(out.shape)
        return out


class OxfordEmbeddingPad(Oxford):
    def load_pc_file(self, filenames, path):
        return pad(loadNumpy(filenames[:-4], path))
        # return torch.load(os.path.join(path, filenames+'.npy')).unsqueeze(0)

    def load_pc_files(self, filenames, path):
        return np.vstack([pad(loadNumpy(f[:-4], path)) for f in filenames])
        # return [torch.load(os.path.join(path, f+'.npy')) for f in filenames]


class OxfordDataModule(LightningDataModule):
    def __init__(self, cfg, data_class=Oxford):
        super().__init__()
        self.cfg = cfg
        self.data_class = data_class

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        data_set = self.data_class(
            query_dict=self.cfg['train_queries'],
            data_dir=self.cfg['data_dir'],
            num_pos=self.cfg['num_positives'],
            num_neg=self.cfg['num_negatives'])

        loader = DataLoader(data_set, batch_size=self.cfg['batch_size'],
                            num_workers=self.cfg['num_worker'], shuffle=True)
        return loader

    def val_dataloader(self):
        data_set = self.data_class(
            query_dict=self.cfg['test_queries'],
            data_dir=self.cfg['data_dir'],
            num_pos=self.cfg['num_positives'],
            num_neg=self.cfg['num_negatives'])

        loader = DataLoader(data_set, batch_size=self.cfg['batch_size'],
                            num_workers=self.cfg['num_worker'])
        return loader

    def test_dataloader(self):
        data_set = self.data_class(
            query_dict=self.cfg['test_queries'],
            data_dir=self.cfg['data_dir'],
            num_pos=self.cfg['num_positives'],
            num_neg=self.cfg['num_negatives'])

        loader = DataLoader(data_set, batch_size=self.cfg['batch_size'],
                            num_workers=self.cfg['num_worker'])
        return loader

    def val_latent_dataloader(self):
        data_set = self.data_class(
            query_dict=self.cfg['test_queries'],
            data_dir=self.cfg['data_dir'],
            num_pos=self.cfg['num_positives'],
            num_neg=self.cfg['num_negatives'],
            return_only_query=True)

        loader = DataLoader(data_set, batch_size=self.cfg['batch_size'],
                            num_workers=self.cfg['num_worker'])
        return loader


######################
####### Test #######
######################

def splitIndex(i, cum_sum):
    # if i < cum_sum[0]:
    #     return 0, i
    smaller = (i < cum_sum[1:])
    bigger = i >= cum_sum[:-1]
    ind = np.argwhere(smaller & bigger).flat[0]
    # print(ind,cum_sum)
    return ind, i-cum_sum[ind]


class OxfordQueryLoader(Dataset):
    def __init__(self, query_dict, data_dir):
        super().__init__()
        with open(query_dict, 'rb') as handle:
            self.query_dict = pickle.load(handle)
        self.nr_scans = [len(d) for d in self.query_dict]
        self.acc_cld = np.array(np.cumsum([0]+self.nr_scans))
        self.dict_keys = [list(d.keys()) for d in self.query_dict]
        self.data_dir = data_dir

    def load_pc_file(self, filenames, path):
        return load_pc_file(filenames, path)

    def __len__(self):
        return self.acc_cld[-1]

    def __getitem__(self, index):
        seq, scan_idx = splitIndex(index, self.acc_cld)
        data = self.query_dict[seq][self.dict_keys[seq][scan_idx]]
        # print(data)
        data['points'] = self.load_pc_file(data['query'], self.data_dir)
        data['seq'] = seq
        data['idx'] = scan_idx
        return data

    def getTruePositives(self, seq, scan, target_seq):
        return self.query_dict[seq][self.dict_keys[seq][scan]][target_seq]

    def getScan(self, seq, scan):
        return self.query_dict[seq][self.dict_keys[seq][scan]]['query']


class OxfordQueryEmbLoaderPad(OxfordQueryLoader):
    def load_pc_file(self, filenames, path):
        return pad(loadNumpy(filenames[:-4], path))
