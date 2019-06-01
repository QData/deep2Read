---
layout: post
title: Basic16- Basic DNN Embedding we read for Ranking/QA/NLP
desc: 2017-team
tags:
- 2Structures
- 9MedApp
categories: 2017Reads
keywords: basic,QA 
---


| Presenter | Papers | Paper URL| Our Slides |
| -----: | -------------------------------------: | :----- | :----- |
| Relation | Translating embeddings for modeling multi-relational data | [PDF](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela) | |
| Relation | A semantic matching energy function for learning with multi-relational data | [PDF](https://link.springer.com/article/10.1007/s10994-013-5363-6)  |   |
| QA | Learning to rank with (a lot of) word features | [PDF](http://ronan.collobert.com/pub/matos/2009_ssi_jir.pdf) |  |
| NLP | A Neural Probabilistic Language Model | [PDF](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  | | 
| NLP | Natural Language Processing (almost) from Scratch | [PDF](https://arxiv.org/abs/1103.0398) |  |
| Metric | ICML07 Best Paper - Information-Theoretic Metric Learning | [PDF](http://www.cs.utexas.edu/users/pjain/pubs/metriclearning_icml.pdf) |  |
| Text | Bag of Tricks for Efficient Text Classification | [PDF](https://arxiv.org/abs/1607.01759) |  |
| QA | Reading wikipedia to answer open-domain questions | [PDF](https://arxiv.org/abs/1704.00051)  | |
| QA | Question answering with subgraph embeddings  | [PDF](https://arxiv.org/abs/1406.3676)  | |
| Train | Curriculum learning [PDF](https://dl.acm.org/citation.cfm?id=1553380) |  | 
| Muthu | NeuroIPS Embedding Papers survey 2012 to 2015| [NIPS](https://papers.nips.cc/) | [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-NIPSEmbedding12to15.pdf) |
| Tobin | Binary embeddings with structured hashed projections [^1] | [PDF](https://arxiv.org/abs/1511.05212) | [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Tobin-BinaryEmbedding.pdf) |


[^1]: <sub><sup> Binary embeddings with structured hashed projections /Anna Choromanska, Krzysztof Choromanski, Mariusz Bojarski, Tony Jebara, Sanjiv Kumar, Yann LeCun (Submitted on 16 Nov 2015 (v1), last revised 1 Jul 2016 (this version, v5))/ We consider the hashing mechanism for constructing binary embeddings, that involves pseudo-random projections followed by nonlinear (sign function) mappings. The pseudo-random projection is described by a matrix, where not all entries are independent random variables but instead a fixed "budget of randomness" is distributed across the matrix. Such matrices can be efficiently stored in sub-quadratic or even linear space, provide reduction in randomness usage (i.e. number of required random values), and very often lead to computational speed ups. We prove several theoretical results showing that projections via various structured matrices followed by nonlinear mappings accurately preserve the angular distance between input high-dimensional vectors. To the best of our knowledge, these results are the first that give theoretical ground for the use of general structured matrices in the nonlinear setting. In particular, they generalize previous extensions of the Johnson-Lindenstrauss lemma and prove the plausibility of the approach that was so far only heuristically confirmed for some special structured matrices. Consequently, we show that many structured matrices can be used as an efficient information compression mechanism. Our findings build a better understanding of certain deep architectures, which contain randomly weighted and untrained layers, and yet achieve high performance on different learning tasks. We empirically verify our theoretical findings and show the dependence of learning via structured hashed projections on the performance of neural network as well as nearest neighbor classifier. </sup></sub>