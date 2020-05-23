# Dual-encoder-with-BERT

## For Quick-Start Experiment in 3sec.
```
git clone https://github.com/izuna385/Dual-encoder-with-BERT.git
cd Dual-encoder-with-BERT
python3 train.py -num_epochs 1
```

For further speednizing, you can use multi gpus.
`CUDA_VISIBLE_DEVICES=0,1 python3 train.py -num_epochs 1 -cuda_devices 0,1`

## Description
Re-implementation of [[Gillick et al., '19]](https://arxiv.org/abs/2004.03555) and [[Humeau et al., '20]](https://arxiv.org/abs/1905.01969) 's bi-encoder.
<div align="center">
<img src="./img/dual_encoder.png" width=70%>
</div>

* You can run Bi-encoder based Entity Linking experiments with your own datasets.

## Notes

* This experiments are specifically for *In-domain* Entity Linking. For *Zero-Shot* one, see [this repository](https://github.com/izuna385/Zero-Shot-Entity-Linking).

[]()
# Requirements
See `requirements.txt`. 
If `allennlp` is not installed to your local environments, follow [Allennlp documentation](https://github.com/allenai/allennlp). 

Also, you need `cui2idx.json`, `idx2cui.json`, `cui2cano.json`, and `cui2def.json` for encoding entities of specified KB (, or, entity set).

* `cui2idx.json` and `idx2json` 

  cui means one unique id for each entity, like `D0002131` of `United stated of America` in DBpedia.
  
  idx is integer for each cui.
  
* `cui2cano.json` and `cui2def.json`

  Canonical names specify entity name for each entity. Canonical names and Definitions (first sentence of definition is often used here) must be split to tokens.

# How to run experiments immediately

* For checking scripts with dummy datasets, run `python3 train.py -num_epochs 1`

  * Linking evaluation is done with entire accuracy, not normalized one.

# How to train Bi-encder with specific datasets?

* Prepare entities mentioned above, and linking dataset.

  * The required formats of datasets can be confirmed at `./dataset/` directory.

# To-do list

* Make dataset creation more easier.

* Pip packaging.

# LICENSE 
MIT
