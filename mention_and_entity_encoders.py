'''
Seq2VecEncoders for encoding mentions and entities.
'''
import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import BertPooler
from overrides import overrides
from allennlp.nn.util import get_text_field_mask, add_positional_features

class Pooler_for_cano_and_def(Seq2VecEncoder):
    def __init__(self, args, word_embedder):
        super(Pooler_for_cano_and_def, self).__init__()
        self.args = args
        self.huggingface_nameloader()
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

    def huggingface_nameloader(self):
        if self.args.bert_name == 'bert-base-uncased':
            self.bert_weight_filepath = 'bert-base-uncased'
        elif self.args.bert_name == 'biobert':
            self.bert_weight_filepath = './biobert_transformers/'
        else:
            self.bert_weight_filepath = 'dummy'
            print('Currently not supported', self.args.bert_name)
            exit()

    def forward(self, cano_and_def_concatnated_text):
        mask_sent = get_text_field_mask(cano_and_def_concatnated_text)
        entity_emb = self.word_embedder(cano_and_def_concatnated_text)
        entity_emb = self.word_embedding_dropout(entity_emb)
        entity_emb = self.bertpooler_sec2vec(entity_emb, mask_sent)

        return entity_emb

class Pooler_for_blink_mention(Seq2VecEncoder):
    def __init__(self, args, word_embedder):
        super(Pooler_for_blink_mention, self).__init__()
        self.args = args
        self.huggingface_nameloader()
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

    def huggingface_nameloader(self):
        if self.args.bert_name == 'bert-base-uncased':
            self.bert_weight_filepath = 'bert-base-uncased'
        elif self.args.bert_name == 'biobert':
            self.bert_weight_filepath = './biobert_transformers/'
        else:
            self.bert_weight_filepath = 'dummy'
            print('Currently not supported', self.args.bert_name)
            exit()

    def forward(self, contextualized_mention):
        mask_sent = get_text_field_mask(contextualized_mention)
        mention_emb = self.word_embedder(contextualized_mention)
        mention_emb = self.word_embedding_dropout(mention_emb)
        mention_emb = self.bertpooler_sec2vec(mention_emb, mask_sent)

        return mention_emb

class Concatenate_Right_and_Left_MentionEncoder(Seq2VecEncoder):
    def __init__(self, args, input_dim, word_embedder):
        super(Concatenate_Right_and_Left_MentionEncoder, self).__init__()
        self.config = args
        self.args = args
        self.input_dim = input_dim

        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)
        self.ff_seq2vecs = nn.Linear(input_dim * 4, input_dim)
        self.huggingface_nameloader()
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)

    def forward(self, left_context_include_mention, right_context_include_mention, mention, context):
        mask_left = get_text_field_mask(left_context_include_mention)
        sentence_left = self.word_embedder(left_context_include_mention)
        sentence_left = self.word_embedding_dropout(sentence_left)
        sentence_left = self.bertpooler_sec2vec(sentence_left, mask_left)

        mask_right = get_text_field_mask(right_context_include_mention)
        sentence_right = self.word_embedder(right_context_include_mention)
        sentence_right = self.word_embedding_dropout(sentence_right)
        sentence_right = self.bertpooler_sec2vec(sentence_right, mask_right)

        mask_ment = get_text_field_mask(mention)
        ment_emb = self.word_embedder(mention)
        ment_emb = self.word_embedding_dropout(ment_emb)
        ment_emb = self.bertpooler_sec2vec(ment_emb, mask_ment)

        mask_sent = get_text_field_mask(context)
        sent_emb = self.word_embedder(context)
        sent_emb = self.word_embedding_dropout(sent_emb)
        sent_emb = self.bertpooler_sec2vec(sent_emb, mask_sent)

        sentence_encoded = torch.cat((sentence_left, sentence_right, ment_emb, sent_emb), dim=1)

        return self.ff_seq2vecs(sentence_encoded)

    @overrides
    def get_output_dim(self):
        return self.input_dim

    def huggingface_nameloader(self):
        if self.args.bert_name == 'bert-base-uncased':
            self.bert_weight_filepath = 'bert-base-uncased'
        elif self.args.bert_name == 'biobert':
            self.bert_weight_filepath = './biobert_transformers/'
        else:
            self.bert_weight_filepath = 'dummy'
            print('Currently not supported', self.args.bert_name)
            exit()

