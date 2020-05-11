import numpy as np
from tqdm import tqdm
import torch
import pdb
from typing import Iterator
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.tokenizers import Token
from utils import OnlyFixedDatasetLoader, KBConstructor_fromKGemb, FixedNegativesEntityLoader
from overrides import overrides
import random
import transformers
from utils import from_original_sentence2left_mention_right_tokens_before_berttokenized

# SEEDS are FIXED
torch.backends.cudnn.deterministic = True
seed = 777
np.random.seed(seed)
torch.manual_seed(seed)

class FixedDatasetTokenizedReader(DatasetReader):
    def __init__(self,args, canonical_and_def_connecttoken, token_indexers=None):
        super().__init__(lazy=args.allen_lazyload)

        self.args = args
        self.max_context_len = args.max_context_len
        self.max_canonical_len = args.max_canonical_len
        self.max_def_len = args.max_def_len

        self.token_indexers = self.token_indexer_returner()
        self.berttokenizer = self.berttokenizer_returner()

        linking_dataset_loader = OnlyFixedDatasetLoader(args=args)
        self.id2line, self.train_mention_id, self.dev_mention_id, self.test_mention_id = linking_dataset_loader.id2line_trn_dev_test_loader()

        print('loading KB')
        self.kbclass = KBConstructor_fromKGemb(args=self.args)
        self.setting_original_KB()
        print('original KB loaded')
        self.ignored_mention_idxs = self.to_be_ignored_mention_idx_checker()
        self.mention_start_token, self.mention_end_token = '[unused1]', '[unused2]'
        self.canonical_and_def_connecttoken = canonical_and_def_connecttoken

    def setting_original_KB(self):
        self.cui2idx, self.idx2cui, self.cui2emb, self.cui2cano, self.cui2def = self.kbclass.return_original_KB()

    def currently_stored_KB_dataset_returner(self):
        return self.cui2idx, self.idx2cui, self.cui2emb, self.cui2cano, self.cui2def

    def huggingfacename_returner(self):
        'Return huggingface modelname and do_lower_case parameter'
        if self.args.bert_name == 'bert-base-uncased':
            return 'bert-base-uncased', True
        elif self.args.bert_name == 'biobert':
            return './biobert_transformers/', False
        else:
            print('Currently',self.args.bert_name,'are not supported.')
            exit()

    def token_indexer_returner(self):
        huggingface_name, do_lower_case = self.huggingfacename_returner()
        return {'tokens': PretrainedTransformerIndexer(
                    model_name=huggingface_name,
                    do_lowercase=do_lower_case)
                }

    def berttokenizer_returner(self):
        if self.args.bert_name == 'bert-base-uncased':
            vocab_file = './vocab_file/bert-base-uncased-vocab.txt'
            do_lower_case = True
        elif self.args.bert_name == 'biobert':
            vocab_file = './vocab_file/biobert_v1.1_pubmed_vocab.txt'
            do_lower_case = False
        else:
            print('currently not supported:', self.args.bert_name)
            raise NotImplementedError
        return transformers.BertTokenizer(vocab_file=vocab_file,
                                          do_lower_case=do_lower_case,
                                          do_basic_tokenize=True,
                                          never_split=['<target>','</target>'])

    def tokenizer_custom(self, txt):
        target_anchors = ['<target>', '</target>']
        original_tokens = txt.split(' ')
        new_tokens = list()

        for token in original_tokens:
            if token in target_anchors:
                new_tokens.append(token)
                continue
            else:
                split_to_subwords = self.berttokenizer.tokenize(token) # token is oneword, split_tokens
                if ['[CLS]'] in  split_to_subwords:
                    split_to_subwords.remove('[CLS]')
                if ['[SEP]'] in  split_to_subwords:
                    split_to_subwords.remove('[SEP]')
                if split_to_subwords == []:
                    new_tokens.append('[UNK]')
                else:
                    new_tokens += split_to_subwords

        return new_tokens

    def mention_and_contexttokenizer_followblinkimplementation(self, txt):
        '''
        Args: sentence with space, including target anchor
            txt:

        Returns: [[CLS], split_sub0, ..., [mention_start], mention, [mention_end], ..., [SEP]]

        '''
        mention_start = '<target>'
        mention_end = '</target>'
        left, mention, right = from_original_sentence2left_mention_right_tokens_before_berttokenized(txt)

        new_tokens = list()
        new_tokens.append('[CLS]')

        if len(left) != 0:
            left_tokens = []
            for one_token in left:
                left_tokens += self.berttokenizer.tokenize(one_token)
            new_tokens += left_tokens[:self.args.max_left_context_len]

        new_tokens.append(self.mention_start_token)
        if len(mention) != 0:
            mention_tokens = []
            for one_token in mention:
                mention_tokens += self.berttokenizer.tokenize(one_token)
            new_tokens += mention_tokens[:self.args.max_mention_len]
        new_tokens.append(self.mention_end_token)

        if len(right) != 0:
            right_tokens = []
            for one_token in right:
                right_tokens += self.berttokenizer.tokenize(one_token)
            new_tokens += right_tokens[:self.args.max_right_context_len]
        new_tokens.append('[SEP]')
        return new_tokens

    def find_anchor(self,split_txt,tobefoundtoken):
        for i, word in enumerate(split_txt):
            if word == tobefoundtoken:
                return i
        return -1

    def left_right_mention_sentence_from_anchorincludedsentence_returner(self, split_txt):
        i = self.find_anchor(split_txt=split_txt, tobefoundtoken='<target>') # mention start
        j = self.find_anchor(split_txt=split_txt, tobefoundtoken='</target>') # mention end

        sfm_mention = split_txt[i+1:j]
        raw_sentence_noanchor = [token for token in split_txt if not token in ['<target>', '</target>']]

        left_context_include_mention = split_txt[:j]
        left_context_include_mention.remove('<target>')
        right_context_include_mention = split_txt[i+1:]
        right_context_include_mention.remove('</target>')

        return raw_sentence_noanchor, sfm_mention, left_context_include_mention, right_context_include_mention

    @overrides
    def _read(self, train_dev_testflag) -> Iterator[Instance]:
        mention_ids = list()
        if train_dev_testflag == 'train':
            mention_ids += self.train_mention_id
            # Because original data is sorted with pmid documents, we have to shuffle data points for in-batch training.
            random.shuffle(mention_ids)
        elif train_dev_testflag == 'dev':
            mention_ids += self.dev_mention_id
        elif train_dev_testflag == 'test':
            mention_ids += self.test_mention_id

        for idx, mention_uniq_id in tqdm(enumerate(mention_ids)):
            if mention_uniq_id in self.ignored_mention_idxs:
                continue
            if self.args.model_for_training == 'blink_implementation_inbatchencoder':
                data = self.linesparser_for_blink_implementation(line=self.id2line[mention_uniq_id], mention_uniq_id=mention_uniq_id)
            else:
                data = self.lineparser_for_local_mentions(line=self.id2line[mention_uniq_id],  mention_uniq_id=mention_uniq_id)
            yield self.text_to_instance(data=data)

    def lineparser_for_local_mentions(self, line, mention_uniq_id):
        '''
        Now this function is going to be depreceated,
        since we gonna follow faithfully with "Zero-shot entity linking with dense entity retrieval"

        Args:
            line:
            train_dev_testflag:
            mention_uniq_id:

        Returns:

        '''
        gold_cui, gold_type, gold_surface_mention, targetanchor_included_sentence = line.split('\t')
        gold_cui = gold_cui.replace('UMLS:', '')
        tokenized_context_including_target_anchors = self.tokenizer_custom(txt=targetanchor_included_sentence)
        raw_sentence_noanchor, sfm_mention, left_context_include_mention, right_context_include_mention = self.left_right_mention_sentence_from_anchorincludedsentence_returner(
            split_txt=tokenized_context_including_target_anchors)

        data = {}

        data['mention_uniq_id'] = mention_uniq_id
        data['gold_ids'] = gold_cui  # str
        data['gold_id_idx_with_cui2idx'] = int(self.cui2idx[gold_cui])
        data['mention_raw'] = gold_surface_mention
        data['raw_sentence_without_anchor_str'] = ' '.join(raw_sentence_noanchor)

        data['context'] = [Token(word) for word in raw_sentence_noanchor][:self.args.max_context_len]
        data['mention_preprocessed'] = [Token(word) for word in sfm_mention][:self.max_context_len]

        if len(left_context_include_mention) <= self.max_context_len:
            data['left_context_include_mention'] = [Token(word) for word in left_context_include_mention]
        else:
            data['left_context_include_mention'] = [Token(word) for word in left_context_include_mention][
                                                   len(left_context_include_mention) - self.max_context_len:]

        data['right_context_include_mention'] = [Token(word) for word in right_context_include_mention][:self.max_context_len]

        data['context'].insert(0, Token('[CLS]'))
        data['context'].insert(len(data['context']), Token('[SEP]'))
        data['mention_preprocessed'].insert(0, Token('[CLS]'))
        data['mention_preprocessed'].insert(len(data['mention_preprocessed']), Token('[SEP]'))
        data['left_context_include_mention'].insert(0, Token('[CLS]'))
        data['left_context_include_mention'].insert(len(data['left_context_include_mention']), Token('[SEP]'))
        data['right_context_include_mention'].insert(0, Token('[CLS]'))
        data['right_context_include_mention'].insert(len(data['right_context_include_mention']), Token('[SEP]'))

        data['gold_cui_cano_and_def_concatenated'] = self.gold_canonical_and_def_concatenated_returner(gold_cui=gold_cui)

        return data

    def linesparser_for_blink_implementation(self, line, mention_uniq_id):
        gold_cui, gold_type, gold_surface_mention, targetanchor_included_sentence = line.split('\t')
        gold_cui = gold_cui.replace('UMLS:', '')
        tokenized_context_including_target_anchors = self.mention_and_contexttokenizer_followblinkimplementation(txt=targetanchor_included_sentence)
        tokenized_context_including_target_anchors = [Token(split_token) for split_token in tokenized_context_including_target_anchors]
        data = {}
        data['context'] = tokenized_context_including_target_anchors
        data['gold_cui_cano_and_def_concatenated'] = self.gold_canonical_and_def_concatenated_returner(gold_cui=gold_cui)
        data['gold_cuidx'] = int(self.cui2idx[gold_cui])
        data['mention_uniq_id'] = int(mention_uniq_id)
        return data

    def gold_canonical_and_def_concatenated_returner(self, gold_cui):
        canonical = self.tokenizer_custom(txt=self.cui2cano[gold_cui])
        definition = self.tokenizer_custom(txt=self.cui2def[gold_cui])

        concatenated = ['[CLS]']
        concatenated += canonical[:self.max_canonical_len]
        concatenated.append(self.canonical_and_def_connecttoken)
        concatenated += definition[:self.max_def_len]
        concatenated.append('[SEP]')

        return [Token(tokenized_word) for tokenized_word in concatenated]

    def to_be_ignored_mention_idx_checker(self):
        to_be_ignored_mention_idxs = []
        all_mention_idxs = list()
        all_mention_idxs += self.train_mention_id
        all_mention_idxs += self.dev_mention_id
        all_mention_idxs += self.test_mention_id
        for mention_idx in all_mention_idxs:
            gold_cui_or_dui = self.id2line[mention_idx].split('\t')[0].replace('UMLS:', '')
            if gold_cui_or_dui not in self.cui2idx:
                to_be_ignored_mention_idxs.append(mention_idx)
        return to_be_ignored_mention_idxs

    @overrides
    def text_to_instance(self, data=None) -> Instance:
        if self.args.model_for_training == 'blink_implementation_inbatchencoder':
            context_field = TextField(data['context'], self.token_indexers)
            fields = {"context": context_field}
            fields['gold_cui_cano_and_def_concatenated'] = TextField(data['gold_cui_cano_and_def_concatenated'], self.token_indexers)
            fields['gold_cuidx'] = ArrayField(np.array(data['gold_cuidx']))
            fields['mention_uniq_id'] = ArrayField(np.array(data['mention_uniq_id']))
        else:
            context_field = TextField(data['context'], self.token_indexers)
            fields = {"context": context_field}
            surface_mention_field = TextField(data['mention_preprocessed'], self.token_indexers)
            fields['left_context_include_mention'] = TextField(data['left_context_include_mention'], self.token_indexers)
            fields['right_context_include_mention'] = TextField(data['right_context_include_mention'], self.token_indexers)
            fields['mention_processed'] = surface_mention_field
            fields['gold_cui_cano_and_def_concatenated'] = TextField(data['gold_cui_cano_and_def_concatenated'], self.token_indexers)
            fields['gold_id_for_knn'] = ArrayField(np.array(data['gold_id_idx_with_cui2idx']))

        return Instance(fields)
'''
For encoding all entities, we need another datasetreader
'''
class AllEntityCanonical_and_Defs_loader(DatasetReader):
    def __init__(self, args, idx2cui, cui2cano, cui2def,
                 textfield_embedder, pretrained_tokenizer, tokenindexer, canonical_and_def_connect_token):
        super().__init__(lazy=args.allen_lazyload)

        self.args = args
        self.idx2cui = idx2cui
        self.cui2cano = cui2cano
        self.cui2def = cui2def
        self.textfield_embedder = textfield_embedder
        self.pretrained_tokenizer = pretrained_tokenizer
        self.token_indexers = tokenindexer
        self.canonical_and_def_connect_token = canonical_and_def_connect_token

    @overrides
    def _read(self,file_path=None) -> Iterator[Instance]:
        for idx, cui in tqdm(self.idx2cui.items()):
            if self.args.debug_for_entity_encoder and idx==2100:
                break
            data = self.cui2data(cui=cui, idx=idx)
            yield self.text_to_instance(data=data)

    @overrides
    def text_to_instance(self, data=None) -> Instance:
        cano_and_def_concatenated = TextField(data['cano_and_def_concatenated'], self.token_indexers)
        fields = {"cano_and_def_concatenated": cano_and_def_concatenated, 'cui_idx':ArrayField(np.array(data['cui_idx'], dtype='int32'))}

        return Instance(fields)

    def tokenizer_custom(self, txt):
        original_tokens = txt.split(' ')
        new_tokens = list()

        for token in original_tokens:
            split_to_subwords = self.pretrained_tokenizer.tokenize(token) # token is oneword, split_tokens
            if ['[CLS]'] in split_to_subwords:
                split_to_subwords.remove('[CLS]')
            if ['[SEP]'] in split_to_subwords:
                split_to_subwords.remove('[SEP]')
            if split_to_subwords == []:
                new_tokens.append('[UNK]')
            else:
                new_tokens += split_to_subwords

        return new_tokens

    def cui2data(self, cui, idx):
        canonical_plus_definition = []
        canonical_plus_definition.append('[CLS]')

        canonical = self.cui2cano[cui]
        canonical_tokens = [Token(split_word) for split_word in self.tokenizer_custom(txt=canonical)]
        canonical_plus_definition += canonical_tokens[:self.args.max_canonical_len]

        canonical_plus_definition.append(self.canonical_and_def_connect_token)

        definition = self.cui2def[cui]
        definition_tokens = [Token(split_word) for split_word in self.tokenizer_custom(txt=definition)]
        canonical_plus_definition += definition_tokens[:self.args.max_def_len]

        canonical_plus_definition.append('[SEP]')

        return {'cano_and_def_concatenated':[Token(split_word_) for split_word_ in canonical_plus_definition], 'cui_idx': idx}