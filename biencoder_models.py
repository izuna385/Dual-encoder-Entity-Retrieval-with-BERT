'''
Model classes
'''
import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.functional import normalize
import torch.nn.functional as F
import copy
import pdb

class InBatchBiencoder(Model):
    def __init__(self, args, input_dim,
                 mention_encoder: Seq2VecEncoder,
                 entity_encoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.output_dim_from_mention_encoder = self.mention_encoder.get_output_dim()
        self.accuracy = CategoricalAccuracy()
        self.istrainflag = 1
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.BCEloss = nn.BCELoss()
        self.cossimloss = nn.CosineEmbeddingLoss()
        self.context2entityemb = nn.Linear(self.output_dim_from_mention_encoder , input_dim)
        self.entity_encoder = entity_encoder
        self.scalingfactor = nn.Parameter(torch.FloatTensor([1.0]).cuda())
        self.mesloss = nn.MSELoss()

    def forward(self, context, left_context_include_mention, right_context_include_mention, mention_processed, gold_cui_cano_and_def_concatenated, gold_id_for_knn):
        batch_num = context['tokens'].size(0)
        contextualized_mention = self.mention_encoder(left_context_include_mention, right_context_include_mention,mention_processed, context)
        contextualized_mention = self.context2entityemb(contextualized_mention)
        encoded_entites = self.entity_encoder(cano_and_def_concatnated_text=gold_cui_cano_and_def_concatenated)

        contextualized_mention = normalize(contextualized_mention, dim=1)
        encoded_entites = normalize(encoded_entites, dim=1)
        # encoded_entites = encoded_entites.unsqueeze(0).repeat(batch_num, 1, 1)

        cossim_dot = torch.matmul(contextualized_mention, encoded_entites.t()) # torch.bmm(encoded_entites, contextualized_mention.unsqueeze(2)).squeeze(2)

        golds = torch.eye(batch_num).cuda()
        # loss = F.log_softmax(cossim_dot * self.scalingfactor, dim=-1) * mask
        # loss = (-loss.sum(dim=1)).mean()

        output = {'loss':self.BCEWloss(cossim_dot * self.scalingfactor, golds)}
        self.accuracy(cossim_dot, torch.argmax(golds, dim=1))
        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def return_entity_encoder(self):
        return self.entity_encoder

class InBatchBLINKBiencoder(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 entity_encoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.accuracy = CategoricalAccuracy()
        self.istrainflag = 1
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.mesloss = nn.MSELoss()
        self.entity_encoder = entity_encoder
        self.cuda_flag = 1

    def forward(self, context, gold_cui_cano_and_def_concatenated, gold_cuidx, mention_uniq_id):
        batch_num = context['tokens'].size(0)
        contextualized_mention = self.mention_encoder(context)
        encoded_entites = self.entity_encoder(cano_and_def_concatnated_text=gold_cui_cano_and_def_concatenated)

        contextualized_mention_forcossim = normalize(contextualized_mention, dim=1)
        encoded_entites_forcossim = normalize(encoded_entites, dim=1)
        # scores = contextualized_mention_forcossim.mm(encoded_entites_forcossim.t())
        scores = contextualized_mention.mm(encoded_entites.t())

        device = torch.get_device(scores) if self.cuda_flag else torch.device('cpu')

        target = torch.LongTensor(torch.arange(batch_num)).to(device)
        loss = F.cross_entropy(scores, target, reduction="mean")

        output = {'loss': loss}
        if self.args.add_mse:
            output['loss'] += 0.01 * self.mesloss(contextualized_mention, encoded_entites)

        if self.istrainflag:
            golds = torch.eye(batch_num).to(device)
            self.accuracy(scores, torch.argmax(golds, dim=1))
        else:
            output['gold_cuidx'] = gold_cuidx
            output['encoded_mentions'] = contextualized_mention
        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def return_entity_encoder(self):
        return self.entity_encoder

    def switch2eval(self):
        self.istrainflag = copy.copy(0)


class BLINKBiencoder_OnlyforEncodingMentions(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder

    def forward(self, context, gold_cui_cano_and_def_concatenated, gold_cuidx, mention_uniq_id):
        contextualized_mention = self.mention_encoder(context)
        output = {'mention_uniq_id': mention_uniq_id,
                  'gold_cuidx': gold_cuidx,
                  'contextualized_mention': contextualized_mention}

        return output

class WrappedModel_for_entityencoding(Model):
    def __init__(self, args,
                 entity_encoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.entity_encoder = entity_encoder

    def forward(self, cui_idx, cano_and_def_concatenated):
        encoded_entites = self.entity_encoder(cano_and_def_concatnated_text=cano_and_def_concatenated)
        output = {'cui_idx': cui_idx, 'emb_of_entities_encoded': encoded_entites}

        return output

    def return_entity_encoder(self):
        return self.entity_encoder

class FixedNegatives_Biencoder(Model):
    def __init__(self, args, input_dim,
                 mention_encoder: Seq2VecEncoder,
                 entity_encoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.output_dim_from_mention_encoder = self.mention_encoder.get_output_dim()
        self.accuracy = CategoricalAccuracy()
        self.istrainflag = 1
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.BCEloss = nn.BCELoss()
        self.cossimloss = nn.CosineEmbeddingLoss()
        self.context2entityemb = nn.Linear(self.output_dim_from_mention_encoder , input_dim)
        self.entity_encoder = entity_encoder
        self.scalingfactor = nn.Parameter(torch.FloatTensor([1.0]).cuda())
        print('NotImplemented Currently')
        exit()
