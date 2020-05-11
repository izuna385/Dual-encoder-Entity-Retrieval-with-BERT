import numpy as np
from tqdm import tqdm
import torch
from allennlp.nn import util as nn_util
from allennlp.data.iterators import BasicIterator
import pdb

class InKBAllEntitiesEncoder:
    def __init__(self, args, entity_loader_datasetreaderclass, entity_encoder_wrapping_model, vocab):
        self.args = args
        self.entity_loader_datasetreader = entity_loader_datasetreaderclass
        self.sequence_iterator_for_encoding_entities = BasicIterator(batch_size=128)
        self.sequence_iterator_for_encoding_entities.index_with(vocab)
        self.entity_encoder_wrapping_model = entity_encoder_wrapping_model
        self.entity_encoder_wrapping_model.eval()
        self.cuda_device = 0

    def encoding_all_entities(self):
        cuidx2emb = {}
        ds = self.entity_loader_datasetreader.read('test')
        entity_generator = self.sequence_iterator_for_encoding_entities(ds, num_epochs=1, shuffle=False)
        entity_generator_tqdm = tqdm(entity_generator, total=self.sequence_iterator_for_encoding_entities.get_num_batches(ds))
        print('======Encoding all entites from canonicals and definitions=====')
        with torch.no_grad():
            for batch in entity_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                cuidxs, embs = self._extract_cuidx_and_its_encoded_emb(batch)
                for cuidx, emb in zip(cuidxs, embs):
                    cuidx2emb.update({int(cuidx):emb})
        return cuidx2emb

    def tonp(self, tsr):
        return tsr.detach().cpu().numpy()

    def _extract_cuidx_and_its_encoded_emb(self, batch) -> np.ndarray:
        out_dict = self.entity_encoder_wrapping_model(**batch)
        return self.tonp(out_dict['cui_idx']), self.tonp(out_dict['emb_of_entities_encoded'])