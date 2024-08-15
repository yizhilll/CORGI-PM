# 介绍

CORGI-PM🐶 是一个中文性别偏见探测和缓解语料库，包含32.9k个句子，通过按照特定的中文性别偏见注释方案进行高质量标注。

我们提出了自动文本性别偏见缓解相关的三个挑战：模型被要求能勾检测、分类和缓解文本中的性别偏见。

# 数据用途

我们的数据集存储在 .npy 二进制文件中，可以轻松打开并转成其他格式。

## 标注「有偏见」的语料

数据集结构如下：

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

加载语料和标注的指令：
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


## 标注「无偏见」的语料

无偏语料库也以 .npy 格式存储，但比较简单。它只有一个 `text` key，因为不需要额外的注释。数据集的结构如下所示：

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

加载数据集的方法：

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

# 自动去除文本性别偏见相关实验


## 偏见检测

我们将偏见检测任务定义为二分类任务，执行以下命令可以复现实验：

```shell
python -u src/run_classification.py detection 
```

## 偏见分类

性别偏见类型分类被定义为多标签分类任务，运行以下命令复现实验：

```shell
python -u src/run_classification.py detection 
```