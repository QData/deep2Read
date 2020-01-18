---
layout: post
title: Basic16- Basic Deep NN with Memory 
desc: 2017-team
categories:
- 0Basics
- 2Architecture
- 7MetaDomain
term: 2017Reads
tags: [ memory, NTM, seq2seq, pointer, set, attention, meta-learning,  Few-Shot, matching net, metric-learning  ]
---


| Presenter | Papers | Paper URL| Our Slides |
| -----: | -------------------------------------: | :----- | :----- |
| seq2seq | Sequence to Sequence Learning with Neural Networks  | [PDF](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural) |  |
| Set | Pointer Networks  | [PDF](https://arxiv.org/abs/1506.03134) |  |
| Set | Order Matters: Sequence to Sequence for Sets | [PDF](https://arxiv.org/abs/1511.06391) |  |
| Point Attention | Multiple Object Recognition with Visual Attention | [PDF](https://arxiv.org/abs/1412.7755) |  |
| Memory | End-To-End Memory Networks  | [PDF](https://arxiv.org/abs/1503.08895) | [Jack Survey]({{site.baseurl}}/MoreTalksTeam/Jack/2016-12-MemoryAttentionModels.pdf) |
| Memory | Neural Turing Machines | [PDF](https://arxiv.org/abs/1410.5401) |  |
| Memory |Hybrid computing using a neural network with dynamic external memory | [PDF](https://www.nature.com/articles/nature20101) |  |
| Muthu | Matching Networks for One Shot Learning (NIPS16) [^1]| [PDF](https://arxiv.org/abs/1606.04080) | [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-MatchingNet.pdf) |
| Jack | Meta-Learning with Memory-Augmented Neural Networks (ICML16) [^2]| [PDF](http://proceedings.mlr.press/v48/santoro16.pdf) | [PDF]({{site.baseurl}}/MoreTalksTeam/Jack/20170724_Memory_Augmented.pdf) |
| Metric | ICML07 Best Paper - Information-Theoretic Metric Learning | [PDF](http://www.cs.utexas.edu/users/pjain/pubs/metriclearning_icml.pdf) |  |

<!--excerpt.start-->

[^1]: <sub><sup> Matching Networks for One Shot Learning (NIPS16): Learning from a few examples remains a key challenge in machine learning. Despite recent advances in important domains such as vision and language, the standard supervised deep learning paradigm does not offer a satisfactory solution for learning new concepts rapidly from little data. In this work, we employ ideas from metric learning based on deep neural features and from recent advances that augment neural networks with external memories. Our framework learns a network that maps a small labelled support set and an unlabelled example to its label, obviating the need for fine-tuning to adapt to new class types. We then define one-shot learning problems on vision (using Omniglot, ImageNet) and language tasks. Our algorithm improves one-shot accuracy on ImageNet from 87.6% to 93.2% and from 88.0% to 93.8% on Omniglot compared to competing approaches. We also demonstrate the usefulness of the same model on language modeling by introducing a one-shot task on the Penn Treebank. </sup></sub>


[^2]: <sub><sup> Meta-Learning with Memory-Augmented Neural Networks (ICML16) Despite recent breakthroughs in the applications of deep neural networks, one setting that presents a persistent challenge is that of "one-shot learning." Traditional gradient-based networks require a lot of data to learn, often through extensive iterative training. When new data is encountered, the models must inefficiently relearn their parameters to adequately incorporate the new information without catastrophic interference. Architectures with augmented memory capacities, such as Neural Turing Machines (NTMs), offer the ability to quickly encode and retrieve new information, and hence can potentially obviate the downsides of conventional models. Here, we demonstrate the ability of a memory-augmented neural network to rapidly assimilate new data, and leverage this data to make accurate predictions after only a few samples. We also introduce a new method for accessing an external memory that focuses on memory content, unlike previous methods that additionally use memory location-based focusing mechanisms. </sup></sub>



 