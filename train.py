# Bi-encoder includes KGemb-retrieval system, too.
import pdb, time, math
from biencoder_parameters import Biencoder_params
from utils import experiment_logger, EmbLoader, ForOnlyFaiss_KBIndexer, cuda_device_parser, parse_cuidx2encoded_emb_for_debugging
from utils import parse_cuidx2encoded_emb_2_cui2emb
from data_reader import FixedDatasetTokenizedReader, AllEntityCanonical_and_Defs_loader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from mention_and_entity_encoders import Concatenate_Right_and_Left_MentionEncoder, Pooler_for_cano_and_def, Pooler_for_blink_mention
from biencoder_models import InBatchBiencoder, InBatchBLINKBiencoder, WrappedModel_for_entityencoding, BLINKBiencoder_OnlyforEncodingMentions
import torch.optim as optim
from KBentities_encoder import InKBAllEntitiesEncoder
from evaluator import BLINKBiEncoderTopXRetriever, DevandTest_BLINKBiEncoder_IterateEvaluator
import copy
import torch
CANONICAL_AND_DEF_CONNECTTOKEN = '[unused3]'

def main():
    exp_start_time = time.time()
    Parameters = Biencoder_params()
    opts = Parameters.get_params()
    experiment_logdir = experiment_logger(args=opts)
    Parameters.dump_params(experiment_dir=experiment_logdir)
    cuda_devices = cuda_device_parser(str_ids=opts.cuda_devices)

    reader_for_mentions = FixedDatasetTokenizedReader(args=opts, canonical_and_def_connecttoken=CANONICAL_AND_DEF_CONNECTTOKEN)
    trains = reader_for_mentions.read('train')
    if not opts.allen_lazyload:
        print('\ntrain statistics:', len(trains), '\n')

    vocab = Vocabulary()
    iterator_for_training_and_evaluating_mentions = BucketIterator(batch_size=opts.batch_size_for_train, sorting_keys=[('context', 'num_tokens')])
    iterator_for_training_and_evaluating_mentions.index_with(vocab)

    embloader = EmbLoader(args=opts)
    emb_mapper, emb_dim, textfieldEmbedder = embloader.emb_returner()

    if opts.model_for_training == 'blink_implementation_inbatchencoder':
        mention_encoder = Pooler_for_blink_mention(args=opts, word_embedder=textfieldEmbedder)
    else:
        mention_encoder = Concatenate_Right_and_Left_MentionEncoder(args=opts, input_dim=emb_dim, word_embedder=textfieldEmbedder)

    current_cui2idx, current_idx2cui, current_cui2emb, current_cui2cano, current_cui2def = reader_for_mentions.currently_stored_KB_dataset_returner()
    fortrainigmodel_faiss_stored_kb_kgemb = ForOnlyFaiss_KBIndexer(args=opts,
                                                             input_cui2idx=current_cui2idx,
                                                             input_idx2cui=current_idx2cui,
                                                             input_cui2emb=current_cui2emb,
                                                             search_method_for_faiss=opts.search_method_before_re_sorting_for_faiss)
    if opts.model_for_training == 'biencoder':
        entity_encoder = Pooler_for_cano_and_def(args=opts, word_embedder=textfieldEmbedder)
        model = InBatchBiencoder(args=opts, mention_encoder=mention_encoder, entity_encoder=entity_encoder, vocab=vocab, input_dim=emb_dim)
    elif opts.model_for_training == 'blink_implementation_inbatchencoder':
        entity_encoder = Pooler_for_cano_and_def(args=opts, word_embedder=textfieldEmbedder)
        model = InBatchBLINKBiencoder(args=opts, mention_encoder=mention_encoder, entity_encoder=entity_encoder, vocab=vocab)
    elif opts.model_for_training == 'fixednegatives_biencoder':
        raise NotImplementedError
    else:
        print('currently', opts.model_for_training,'are not supported')
        raise NotImplementedError
    model = model.cuda()

    if not opts.debug_for_entity_encoder:
        optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=opts.lr,eps=opts.epsilon,
                               weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2), amsgrad=opts.amsgrad)
        trainer = Trainer(model=model,optimizer=optimizer,
                          iterator=iterator_for_training_and_evaluating_mentions, train_dataset=trains,
                          # validation_dataset=devs,
                          cuda_device=cuda_devices, num_epochs=opts.num_epochs
                          )
        trainer.train()
    else:
        print('\n==Skip Biencoder training==\n')

    with torch.no_grad():
        model.eval()
        model.switch2eval()

        print('======Start encoding all entities in KB=====\n======1. Start Tokenizing All Entities=====')
        entity_encoder_wrapping_model = WrappedModel_for_entityencoding(args=opts, entity_encoder=entity_encoder, vocab=vocab)
        entity_encoder_wrapping_model.eval()
        # entity_encoder_wrapping_model.cpu()

        Tokenizer = reader_for_mentions.berttokenizer_returner()
        TokenIndexer = reader_for_mentions.token_indexer_returner()
        kbentity_loader = AllEntityCanonical_and_Defs_loader(args=opts, idx2cui=current_idx2cui, cui2cano=current_cui2cano,
                                                             cui2def=current_cui2def, textfield_embedder=textfieldEmbedder,
                                                             pretrained_tokenizer=Tokenizer, tokenindexer=TokenIndexer,
                                                             canonical_and_def_connect_token=CANONICAL_AND_DEF_CONNECTTOKEN)
        Allentity_embedding_encodeIterator = InKBAllEntitiesEncoder(args=opts, entity_loader_datasetreaderclass=kbentity_loader,
                                                                    entity_encoder_wrapping_model=entity_encoder_wrapping_model,
                                                                    vocab=vocab)
        print('======2. Encoding All Entities=====')
        cuidx2encoded_emb = Allentity_embedding_encodeIterator.encoding_all_entities()
        if opts.debug_for_entity_encoder:
            cuidx2encoded_emb = parse_cuidx2encoded_emb_for_debugging(cuidx2encoded_emb=cuidx2encoded_emb, original_cui2idx=current_cui2idx)
        cui2encoded_emb = parse_cuidx2encoded_emb_2_cui2emb(cuidx2encoded_emb=cuidx2encoded_emb, original_cui2idx=current_cui2idx)
        print('=====Encoding all entities in KB FINISHED!=====')

        print('\n+++++Indexnizing KB from encoded entites+++++')
        forstoring_encoded_entities_to_faiss = ForOnlyFaiss_KBIndexer(args=opts,
                                                                 input_cui2idx=current_cui2idx,
                                                                 input_idx2cui=current_idx2cui,
                                                                 input_cui2emb=cui2encoded_emb,
                                                                 search_method_for_faiss=opts.search_method_before_re_sorting_for_faiss,
                                                                       entity_emb_dim=768)
        print('+++++Indexnizing KB from encoded entites FINISHED!+++++')

        print('Loading BLINKBiencoder')
        blinkbiencoder_onlyfor_encodingmentions = BLINKBiencoder_OnlyforEncodingMentions(args=opts, mention_encoder=mention_encoder, vocab=vocab)
        blinkbiencoder_onlyfor_encodingmentions.cuda()
        blinkbiencoder_onlyfor_encodingmentions.eval()
        print('Loaded: BLINKBiencoder')

        print('Evaluation for BLINK start')
        blinkBiEncoderEvaluator = BLINKBiEncoderTopXRetriever(args=opts, vocab=vocab, blinkbiencoder_onlyfor_encodingmentions=blinkbiencoder_onlyfor_encodingmentions,
                                                          fortrainigmodel_faiss_stored_kb_kgemb=forstoring_encoded_entities_to_faiss.indexed_faiss_returner(),
                                                          reader_for_mentions=reader_for_mentions)
        finalblinkEvaluator = DevandTest_BLINKBiEncoder_IterateEvaluator(args=opts, blinkBiEncoderEvaluator=blinkBiEncoderEvaluator, experiment_logdir=experiment_logdir)
        finalblinkEvaluator.final_evaluation(dev_or_test_flag='dev')
        finalblinkEvaluator.final_evaluation(dev_or_test_flag='test')

        exp_end_time = time.time()
        print('Experiment time', math.floor(exp_end_time - exp_start_time), 'sec')

if __name__ == '__main__':
    main()
