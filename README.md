# L2Copy4APE
## Contents
* [Introduction](#Introduction)
* [Prerequisites](#Prerequisites)
* [Datasets](#Datasets)
* [Preprocess](#Preprocess)
* [Training](#Training)
* [Inference](#Inference)
* [Hyper-params](#Hyper-params)
* [Evaluation](#Evaluation)
* [Download](#Download)
* [Citation](#Citation)
* [Contact](#Contact)


## Introduction
This is the code for the paper "[Learning to Copy for Automatic Post-Editing](https://www.aclweb.org/anthology/D19-1634/)" (EMNLP2019).  The implementation is on top of the open-source NMT toolkit [THUMT](https://github.com/thumt/THUMT). You might need to glance over the user manual of THUMT for knowing the basic usage of THUMT.

## Prerequisites
* Python 2.7
* Tensorflow 1.11

## Datasets
* [WMT 2016-2018 datasets](http://statmt.org/wmt18/ape-task.html)

Training, development and test data consist English-German triplets (source, target, and post-edit) belonging to the IT domain, and are already tokenized. 

## Preprocess
We apply Truecasing ([train-truecaser.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/train-truecaser.perl), [truecase.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/truecase.perl)) and BPE ([learn_joint_bpe_and_vocab.py](https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/learn_joint_bpe_and_vocab.py), [apply_bpe.py](https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/apply_bpe.py)) to these datasets.
* learn truecase model
`perl train-truecaser.perl --model $true_model --corpus $corpus`
* apply truecase
`perl truecase.perl -model $true_model < ${corpus} > ${corpus}.tc`
* learn bpe
`python learn_joint_bpe_and_vocab.py --input $src $trg -s 32000 -o bpe32k --write-vocabulary vocab.$L1 vocab.$L2`
* apply bpe
`python apply_bpe.py --vocabulary vocab.$L1 --vocabulary-threshold 50 -c bpe32k < ${corpus} > ${corpus}.bpe`

For the PBSMT dataset:
1. learn truecase model and BPE from WMT16train + WMT17train + artificial-small (549k)
2. apply truecasing and BPE to the 549k dataset
3. 5M dataset = (WMT16train + WMT17train)\*20 + artificial-small + artificial-big
4. 12M dataset = 5M dataset + eSCAPE-PBSMT
5. apply truecasing and BPE to these datasets
6. generate *copying labels* file using [LCS algorithm](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem)

Note that the eSCAPE-PBSMT dataset need to be tokenized by [tokenizer.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) `tokenizer.perl -no-escape -l $lang -threads 10 -lines 10000 -time < ${corpus} > ${corpus}.tok`.

For the NMT dataset:
1. learn truecase model and BPE from WMT18train + artificial-small (539k)
2. apply truecasing and BPE to the 539k dataset
3. 5M dataset = WMT18train\*20 + artificial-small + artificial-big
4. apply truecasing and BPE to the 5M dataset
5. generate *copying labels* file using [LCS algorithm](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem)

Finally, shuffle these datasets and generate joint vocabulary.

## Training
We employ two-step training:

1. Train with *interactive*+*CopyNet*:
```
PYTHONPATH=/path/to/L2Copy4APE/ \
python /path/to/L2Copy4APE/thumt/bin/trainer.py \
    --input ${train_src} ${train_mt} ${train_pe} ${train_label} \
    --vocabulary ${vocab_joint} ${vocab_joint} \
    --model transformer \
    --validation ${dev_src} \
    --references ${dev_mt} ${dev_pe} ${dev_label} \
    --parameters eval_steps=500,constant_batch_size=false,batch_size=6250,train_steps=50000,save_checkpoint_steps=9999999,shared_embedding_and_softmax_weights=true,device_list=[0,1,2,3],decode_alpha=1.0,eval_batch_size=64,update_cycle=1,keep_top_checkpoint_max=10,shared_source_target_embedding=true,keep_checkpoint_max=1,enable_tagger=False
mkdir -p average
python /path/to/L2Copy4APE/thumt/scripts/checkpoint_averaging.py --path train/eval --output average --checkpoints 10
```
Save the model after model averaging under `pretrained/average`.

2. Train with *interactive*+*CopyNet*+*Predictor*+*Joint Training*:
```
PYTHONPATH=/path/to/L2Copy4APE/ \
python /path/to/L2Copy4APE/thumt/bin/trainer.py \
    --input ${train_src} ${train_mt} ${train_pe} ${train_label} \
    --vocabulary ${vocab_joint} ${vocab_joint} \
    --model transformer \
    --validation ${dev_src} \
    --references ${dev_mt} ${dev_pe} ${dev_label} \
    --checkpoint /path/to/pretrained/average \
    --parameters eval_steps=500,constant_batch_size=false,batch_size=6250,train_steps=50000,save_checkpoint_steps=9999999,shared_embedding_and_softmax_weights=true,device_list=[0,1,2,3],decode_alpha=1.0,eval_batch_size=64,update_cycle=2,keep_top_checkpoint_max=10,shared_source_target_embedding=true,keep_checkpoint_max=1,multi_task_alpha=0.9,where_to_apply='enc',copy_lambda=1.0,enable_tagger=True
mkdir -p average
python /path/to/L2Copy4APE/thumt/scripts/checkpoint_averaging.py --path train/eval --output average --checkpoints 10
```
Save the model after model averaging under `final/average`.

Note that pre-training seems to be useless on 5M NMT dataset, we suggust to directly train the model like step-2 on 5M NMT dataset.

## Inference
```
PYTHONPATH=/path/to/L2Copy4APE/ \
python /path/to/L2Copy4APE/thumt/bin/translator.py \
    --input $src $mt \
    --output $trg \
    --vocabulary ${vocab_joint} ${vocab_joint} \
    --checkpoints /path/to/final/average \
    --model transformer \
    --parameters device_list=[$devs],decode_alpha=1.0,decode_batch_size=64,shared_embedding_and_softmax_weights=true,shared_source_target_embedding=true,where_to_apply='enc',enable_tagger=True
sed -r 's/(@@ )|(@@ ?$)//g' < ${trg} > ${trg}.rbpe
perl ${moses_dir}/scripts/recaser/detruecase.perl < ${trg}.rbpe > ${trg}.rbpe.detc
```

## Hyper-params
* enable_tagger: whether to use *Predictor*+*Joint Training*.
* where_to_apply: where to apply copying scores, you can choose 'enc' for encoder, 'dec' for decoder and CopyNet, and 'both' for all.
* multi_task_alpha: \alpha in the paper.
* copy_lambda: \lambda in the paper.


## Evaluation 
* BLEU-4 ([multi-bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl))
* TER ([tercom](http://www.cs.umd.edu/~snover/tercom/), [download](https://www.dropbox.com/s/5jw5maariwey080/Evaluation_Script.tar.gz?dl=0))

## Download
We provide the pre-trained models, you can skip the training process and do inference directly.
* [[Google Drive](https://drive.google.com/file/d/1QGJ7rmPcbnffMiUdRKBGW86-otjmhnsG/view?usp=sharing)] For 5M training dataset, you should get 72.76/72.55/71.59(DEV16/TEST16/TEST17 BLEU) and 18.61/18.38/18.78(DEV16/TEST16/TEST17 TER). 
* [[Google Drive](https://drive.google.com/file/d/1jcBya4V3kcmy67OQ0kdrSIFxxQdf-UM5/view?usp=sharing)] For 12M training dataset, you should get 73.85/73.83/72.69(DEV16/TEST16/TEST17 BLEU) and 17.78/17.25/17.77(DEV16/TEST16/TEST17 TER). 
## Citation
If you use our codes, please cite our paper:
```
@inproceedings{huang-etal-2019-learning,
    title = "Learning to Copy for Automatic Post-Editing",
    author = "Huang, Xuancheng  and
      Liu, Yang  and
      Luan, Huanbo  and
      Xu, Jingfang  and
      Sun, Maosong",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1634",
    doi = "10.18653/v1/D19-1634",
    pages = "6124--6134",
    abstract = "Automatic post-editing (APE), which aims to correct errors in the output of machine translation systems in a post-processing step, is an important task in natural language processing. While recent work has achieved considerable performance gains by using neural networks, how to model the copying mechanism for APE remains a challenge. In this work, we propose a new method for modeling copying for APE. To better identify translation errors, our method learns the representations of source sentences and system outputs in an interactive way. These representations are used to explicitly indicate which words in the system outputs should be copied. Finally, CopyNet (Gu et al., 2016) can be combined with our method to place the copied words in correct positions in post-edited translations. Experiments on the datasets of the WMT 2016-2017 APE shared tasks show that our approach outperforms all best published results.",
}
```
## Contact
If you have questions, suggestions and bug reports, please email hxc17@mails.tsinghua.edu.cn.
