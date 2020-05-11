import os, json, time, gc, copy, shutil, random, pickle, sys, pdb
from datetime import datetime
import numpy as np
from allennlp.common.tee_logger import TeeLogger
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from pytz import timezone
import faiss
import torch
import torch.nn as nn
from tqdm import tqdm

def cuda_device_parser(str_ids):
    return [int(stridx) for stridx in str_ids.strip().split(',')]

def from_original_sentence2left_mention_right_tokens_before_berttokenized(sentence):
    mention_start = '<target>'
    mention_end = '</target>'
    original_tokens = sentence.split(' ')
    mention_start_idx = int(original_tokens.index(mention_start))
    mention_end_idx = int(original_tokens.index(mention_end))
    if mention_end_idx == len(sentence) - 1 :
        return original_tokens[:mention_start_idx], original_tokens[mention_start_idx+1:mention_end_idx], []
    else:
        return original_tokens[:mention_start_idx], original_tokens[mention_start_idx+1:mention_end_idx], original_tokens[mention_end_idx+1:]

def parse_cuidx2encoded_emb_for_debugging(cuidx2encoded_emb, original_cui2idx):
    print('/////Some entities embs are randomized for debugging./////')
    for cuidx in tqdm(original_cui2idx.values()):
        if cuidx not in cuidx2encoded_emb:
            cuidx2encoded_emb.update({cuidx:np.random.randn(*cuidx2encoded_emb[0].shape)})
    return cuidx2encoded_emb

def parse_cuidx2encoded_emb_2_cui2emb(cuidx2encoded_emb, original_cui2idx):
    cui2emb = {}
    for cui, idx in original_cui2idx.items():
        cui2emb.update({cui:cuidx2encoded_emb[idx]})
    return cui2emb

def experiment_logger(args):
    '''
    :param args: from biencoder_parameters
    :return: dirs for experiment log
    '''
    experimet_logdir = args.experiment_logdir # / is included

    timestamp = datetime.now(timezone('Asia/Tokyo'))
    str_timestamp = '{0:%Y%m%d_%H%M%S}'.format(timestamp)[2:]

    dir_for_each_experiment = experimet_logdir + str_timestamp

    if os.path.exists(dir_for_each_experiment):
        dir_for_each_experiment += '_d'

    dir_for_each_experiment += '/'
    logger_path = dir_for_each_experiment + 'teelog.log'
    os.mkdir(dir_for_each_experiment)

    if not args.debug:
        sys.stdout = TeeLogger(logger_path, sys.stdout, False)  # default: False
        sys.stderr = TeeLogger(logger_path, sys.stderr, False)  # default: False

    return dir_for_each_experiment

def from_jsonpath_2_str2idx(json_path):
    str2intidx = {}
    with open(json_path, 'r') as f:
        tmp_str2stridx = json.load(f)
    for str_key, str_idx in tmp_str2stridx.items():
        str2intidx.update({str_key:int(str_idx)})
    return str2intidx

def from_jsonpath_2_idx2str(json_path):
    intidx2str = {}
    with open(json_path, 'r') as f:
        tmp_stridx2str = json.load(f)
    for str_idx, value_str in tmp_stridx2str.items():
        intidx2str.update({int(str_idx):value_str})
    return intidx2str

def from_jsonpath_2_str2strorlist(json_path):
    with open(json_path, 'r') as f:
        raw_json = json.load(f)
    return raw_json

def pklloader(pkl_path):
    with open(pkl_path, 'rb') as p:
        loaded = pickle.load(p)
    return loaded

class EmbLoader:
    def __init__(self, args):
        self.args = args

    def emb_returner(self):
        if self.args.bert_name == 'bert-base-uncased':
            huggingface_model = 'bert-base-uncased'
        elif self.args.bert_name == 'biobert':
            assert self.args.ifbert_use_whichmodel == 'biobert'
            huggingface_model = './biobert_transformers/'
        else:
            huggingface_model = 'dummy'
            print(self.args.bert_name,'are not supported')
            exit()
        bert_embedder = PretrainedTransformerEmbedder(model_name=huggingface_model)
        return bert_embedder, bert_embedder.get_output_dim(), BasicTextFieldEmbedder({'tokens': bert_embedder},
                                                                                     allow_unmatched_keys=True) 

class OnlyFixedDatasetLoader:
    '''
    Before running this, we assume that preprocess has been already done
    '''

    def __init__(self, args):
        self.args = args
        self.dataset = self.dataset_name_returner()
        self.dataset_dir = self.dataset_dir_returner()

    def dataset_name_returner(self):
        assert self.args.dataset in ['xxx', 'yyy', 'zzz']
        return self.args.dataset

    def dataset_dir_returner(self):
        return self.args.mention_dump_dir + self.dataset + '/'

    def fixed_idxnized_datapath_returner(self):
        id2line_json_path = self.dataset_dir + 'id2line.json'
        # pmid2int_mention_path = self.dataset_dir + 'pmid2ment.json'
        train_mentionidpath = self.dataset_dir + 'train_mentionid.pkl'
        dev_mentionidpath = self.dataset_dir + 'dev_mentionid.pkl'
        test_mentionidpath = self.dataset_dir + 'test_mentionid.pkl'

        return id2line_json_path, train_mentionidpath, dev_mentionidpath, test_mentionidpath

    def id2line_path_2_intid2line(self, id2line_json_path):
        with open(id2line_json_path, 'r') as id2l:
            tmp_id2l = json.load(id2l)
        intid2line = {}
        for str_idx, line_mention in tmp_id2l.items():
            intid2line.update({int(str_idx): line_mention})

        return intid2line

    def train_dev_test_mentionid_returner(self, train_mentionidpath, dev_mentionidpath, test_mentionidpath):
        with open(train_mentionidpath, 'rb') as trp:
            train_mentionid = pickle.load(trp)

        with open(dev_mentionidpath, 'rb') as drp:
            dev_mentionid = pickle.load(drp)

        with open(test_mentionidpath, 'rb') as terp:
            test_mentionid = pickle.load(terp)

        if self.args.debug:
            return train_mentionid[:300], dev_mentionid[:200], test_mentionid[:400]
        else:
            return train_mentionid, dev_mentionid, test_mentionid

    def id2line_trn_dev_test_loader(self):
        id2line_json_path, train_mentionidpath, dev_mentionidpath, test_mentionidpath = self.fixed_idxnized_datapath_returner()

        id2line = self.id2line_path_2_intid2line(id2line_json_path=id2line_json_path)
        train_mentionid, dev_mentionid, test_mentionid = self.train_dev_test_mentionid_returner(
            train_mentionidpath=train_mentionidpath,
            dev_mentionidpath=dev_mentionidpath,
            test_mentionidpath=test_mentionidpath)

        return id2line, train_mentionid, dev_mentionid, test_mentionid

class KBConstructor_fromKGemb:
    def __init__(self, args):
        self.args = args
        self.kbemb_dim = self.args.kbemb_dim
        self.original_kbloader_to_memory()

    def original_kbloader_to_memory(self):
        cui2idx_path, idx2cui_path, cui2emb_path, cui2cano_path, cui2def_path = self.from_datasetname_return_related_dicts_paths()
        print('set value and load original KB')
        self.original_cui2idx = from_jsonpath_2_str2idx(cui2idx_path)
        self.original_idx2cui = from_jsonpath_2_idx2str(idx2cui_path)
        self.original_cui2emb = pklloader(cui2emb_path)
        self.original_cui2cano = from_jsonpath_2_str2strorlist(cui2cano_path)
        self.original_cui2def = from_jsonpath_2_str2strorlist(cui2def_path)

    def return_original_KB(self):
        return self.original_cui2idx, self.original_idx2cui, self.original_cui2emb, self.original_cui2cano, self.original_cui2def

    def from_datasetname_return_related_dicts_paths(self):
        assert self.args.dataset in ['xxx','yyy','zzz']

        if self.args.dataset in ['xxx', 'yyy']:
            cui2idx_path = './../src/cui2idx.json'
            idx2cui_path = './../src/idx2cui.json'
            cui2emb_path = './../src/cui2emb.pkl'
            cui2cano_path = './../src/cui2cano.json'
            cui2def_path = './../src/cui2def.json'

        elif self.args.dataset in 'zzz':
            cui2idx_path = './../src/cui2idx.json'
            idx2cui_path = './../src/idx2cui.json'
            cui2emb_path = './../src/cui2emb.pkl'
            cui2cano_path = './../src/cui2cano.json'
            cui2def_path = './../src/cui2def.json'

        else:
            cui2idx_path, idx2cui_path, cui2emb_path, cui2cano_path, cui2def_path = ['dummy' for i in range(5)]
            print(self.args.dataset, 'are currently not supported')
            exit()

        return cui2idx_path, idx2cui_path, cui2emb_path, cui2cano_path, cui2def_path

    def load_original_KBmatrix_alignedwith_idx2cui(self):
        KBemb = np.random.randn(len(self.original_cui2emb.keys()), self.kbemb_dim).astype('float32')

        for idx, cui in self.original_idx2cui.items():
            KBemb[idx] = self.original_cui2emb[cui]

        return KBemb

    def indexed_faiss_loader_for_constructing_smallKB(self):
        if self.args.search_method_for_faiss_during_construct_smallKBfortrain == 'indexflatl2':  # L2
            self.indexed_faiss = faiss.IndexFlatL2(self.kbemb_dim)
        elif self.args.search_method_for_faiss_during_construct_smallKBfortrain == 'indexflatip':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        elif self.args.search_method_for_faiss_during_construct_smallKBfortrain == 'cossim':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        else:
            print('currently',self.args.search_method_for_faiss_during_construct_smallKBfortrain, 'are not supported')
            exit()

        return self.indexed_faiss

class ForOnlyFaiss_KBIndexer:
    def __init__(self, args, input_cui2idx, input_idx2cui, input_cui2emb, search_method_for_faiss, entity_emb_dim=300):
        self.args = args
        self.kbemb_dim = entity_emb_dim
        self.cui2idx = input_cui2idx
        self.idx2cui = input_idx2cui
        self.cui2emb = input_cui2emb
        self.search_method_for_faiss = search_method_for_faiss
        self.indexed_faiss_loader()
        self.KBmatrix = self.KBmatrixloader()
        self.entity_num = len(input_cui2idx)
        self.indexed_faiss_KBemb_adder(KBmatrix=self.KBmatrix)

    def KBmatrixloader(self):
        KBemb = np.random.randn(len(self.cui2idx.keys()), self.kbemb_dim).astype('float32')
        for idx, cui in self.idx2cui.items():
            KBemb[idx] = self.cui2emb[cui]

        return KBemb

    def indexed_faiss_loader(self):
        if self.search_method_for_faiss == 'indexflatl2':  # L2
            self.indexed_faiss = faiss.IndexFlatL2(self.kbemb_dim)
        elif self.search_method_for_faiss == 'indexflatip':  #
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        elif self.search_method_for_faiss == 'cossim':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)

    def indexed_faiss_KBemb_adder(self, KBmatrix):
        if self.search_method_for_faiss == 'cossim':
            KBemb_normalized_for_cossimonly = np.random.randn(self.entity_num, self.kbemb_dim).astype('float32')
            for idx, emb in enumerate(KBmatrix):
                if np.linalg.norm(emb, ord=2, axis=0) != 0:
                    KBemb_normalized_for_cossimonly[idx] = emb / np.linalg.norm(emb, ord=2, axis=0)
            self.indexed_faiss.add(KBemb_normalized_for_cossimonly)
        else:
            self.indexed_faiss.add(KBmatrix)

    def indexed_faiss_returner(self):
        return self.indexed_faiss

    def KBembeddings_loader(self):
        KBembeddings = nn.Embedding(self.entity_num, self.kbemb_dim, padding_idx=0)
        KBembeddings.weight.data.copy_(torch.from_numpy(self.KBmatrix))
        KBembeddings.weight.requires_grad = False
        return KBembeddings

class FixedNegativesEntityLoader:
    def __init__(self, args):
        self.args = args
        self.dataset = self.args.dataset
        self.dataset_checker()

    def train_mention_idx2negatives_loader(self):
        negatives_json_path = './duringtrain_sampled_negs/' + self.args.dataset + '.json'
        with open(negatives_json_path, 'r') as f:
            train_mention_stridx2negativesdata = json.load(f)

        train_mention_intidx2negatives = {}
        for stridx, itsdata in train_mention_stridx2negativesdata.items():
            train_mention_intidx2negatives.update({int(stridx):itsdata})

        return train_mention_intidx2negatives

    def dataset_checker(self):
        try:
            assert self.args.dataset in ['xxx', 'yyy', 'zzz']
        except:
            raise NotImplementedError
