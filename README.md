README languages/语言: 
- [![English](https://img.shields.io/badge/lang-EN-green.svg)](https://github.com/yizhilll/CORGI-PM/READM.md)
- [![中文](https://img.shields.io/badge/lang-中文-red.svg)](https://github.com/yizhilll/CORGI-PM/README.zh.md)

# Introduction

**CORGI-PM **  is a Chinese cOrpus foR Gender bIas Probing and Mitigation, which contains **32.9k** sentences with high-quality labels derived by following an annotation scheme specifically developed for gender bias in the Chinese context.

We address three challenges for automatic textual gender bias mitigation, which requires the models to detect, classify, and mitigate textual gender bias.

# Data Usage

Our dataset is stored in `.npy` binary files and can be easily retrieved.

## Biased Corpus

The dataset is structured as followed:

```python
{
    'train':{
        # original corpus
        'ori_sentence': [
            sent_0,
            sent_1,
            ...,
        ], 
        # bias types, stored as one-hot labels
        'bias_labels': [
            [0 1 0],
            [0 1 0],
            [0 1 0],
            ...,
        ],
        # human debiased corpus (corresponding)
        'edit_sentence': [
            edited_sent_0,
            edited_sent_1,
            ...,
        ],
    },
    'valid':{
        ... # similar
    },
    'test':{
        ... # similar
    }
}
```

Instructions to load the annotated biased corpus:

```python
>>> import numpy as np

# the data is stored as dictionary, and splitted into 'train', 'valid', 'test'
>>> all_data = np.load('dataset/CORGI-PC_splitted_biased_corpus_v1.npy',allow_pickle=True).item()
>>> print(all_data.keys())
dict_keys(['train', 'valid', 'test'])

# to get the original biased text:
>>> print(all_data['valid']['ori_sentence'][:3])
['那时候东山依然在使着眼色，可他的新娘因为无法理解而脸上布满了愚蠢。于是东山便凑过去咬牙切齿地说了一句什么，总算明白过来的新娘脸上出现了幽默的微笑。随即东山和他的新娘一起站了起来。东山站起来时十分粗鲁，他踢倒了椅子。正如森林事先预料的一样，他们走进了那个房间。但是他们没有将门关上，所以森林仍然看到那张床的一只角，不过没有看到他们两人，他们在床的另一端。然后那扇门关上了。不久之后，那间屋子里升起了一种...'
 '下贱东西，大约她知道自己太不行，必须找个比她再下贱的。'
 '胡文玉不只生的魁伟俊秀，而且工作上有魄力，有办法，写得一手好文章，讲起话来又头头是道。']

# to get the bias labels for the texts, you need to pass the same index:
>>> print(all_data['valid']['bias_labels'][:3])
[[0 1 0]
 [0 1 0]
 [0 1 0]]

# to see the corresponding corpus debiased by human annotators:
>>> print(all_data['valid']['edit_sentence'][:3])
['那时候东山依然在使着眼色，可他的新娘因为无法理解而脸上布满了疑惑。于是东山便凑过去咬牙切齿地说了一句什么，总算明白过来的新娘脸上出现了幽默的微笑。随即东山和他的新娘一起站了起来。东山站起来时十分鲁莽，他踢倒了椅子。正如森林事先预料的一样，他们走进了那个房间。但是他们没有将门关上，所以森林仍然看到那张床的一只角，不过没有看到他们两人，他们在床的另一端。然后那扇门关上了。不久之后，那间屋子里升起了一种...'
 '糟糕东西，大约她知道自己太不行，必须找个比她再糟糕的。' '胡文玉不只生的俊秀，而且工作上有魄力，有办法，写得一手好文章，讲起话来又头头是道。']
```


## Non-Biased Corpus

The non-biased corpus is also stored as `.npy` but much simpler. It only has `text` key since it doesn't require extra annotation. The dataset is structured as followed:

```python
{
    'train':{
        # original corpus
        'text': [
            sent_0,
            sent_1,
            ...,
        ], 
    },
    'valid':{
        ... # similar
    },
    'test':{
        ... # similar
    }
}
```

To load the dataset:

```python
>>> import numpy as np
>>> non_bias_corpus = np.load('dataset/CORGI-PC_splitted_non-bias_corpus_v1.npy',allow_pickle=True).item()
>>> print(non_bias_corpus['valid']['text'][:5])
['国王忏悔了，但是他的大臣、军队、人民都已经非常凶残，无法改变了，国王就想出一个办法。', 
'北京队的攻手非常有实力，身高、力量都很好，训练中也安排了男教练进行模仿，在拦防环节要适应更多的重球。', 
'年,她在淘宝开出了一家鹅肝专卖店。', 
'该公司老板表示,当时她感觉到了不对劲,于是就下楼查看,才发现隔壁药店着火了。', 
'那个辛苦劲儿，就是个壮实的男劳力也吃不消，不过我也挺过来了！']
```

# Automatic Textual Gender Bias Mitigation Experiments

## Bias Detection

We formulate the bias detection tasks as binary classification. To run the codes:

```shell
python -u src/run_classification.py detection 
```

## Bias Classification

Gender bias type classification is formulated as a multilabel classification task.

```
python -u src/run_classification.py multilabel  
```

## Bias Mitigation

TBD

# Citation


```bibtex
@misc{https://doi.org/10.48550/arxiv.2301.00395,
  doi = {10.48550/ARXIV.2301.00395},
  url = {https://arxiv.org/abs/2301.00395},
  author = {Zhang, Ge and Li, Yizhi and Wu, Yaoyao and Zhang, Linyuan and Lin, Chenghua and Geng, Jiayi and Wang, Shi and Fu, Jie},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Computers and Society (cs.CY), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {CORGI-PM: A Chinese Corpus For Gender Bias Probing and Mitigation},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
