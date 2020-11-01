---
toc: true
author_profile: true
sidebar:
  title: "Reviews Indexed"
  nav: sidebar-sample
layout: single
title: Application18- Property of DeepNN Models and Discrete tasks 
desc: 2018-team
term: 2018Reads
categories:
- 3Reliable
tags: [ embedding, generative, NLP, generalization, NLP ]
---


| Presenter | Papers | Paper URL| Our Slides |
| -----: | ---------------------------: | :----- | :----- |
| Bill | Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation [^1]| [PDF](https://arxiv.org/abs/1609.08144) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.02.09_GoogleNMT.pdf) | 
| Bill |  Measuring the tendency of CNNs to Learn Surface Statistical Regularities Jason Jo, Yoshua Bengio | [PDF](https://arxiv.org/abs/1711.11561) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.02.16_CNNRegularity.pdf) | 
| Bill | Generating Sentences by Editing Prototypes, Kelvin Guu, Tatsunori B. Hashimoto, Yonatan Oren, Percy Liang [^2]  | [PDF](https://arxiv.org/abs/1709.08878) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.01.25_PrototypeEdit.pdf) | 
| Bill | On the importance of single directions for generalization, Ari S. Morcos, David G.T. Barrett, Neil C. Rabinowitz, Matthew Botvinick  | [PDF](https://arxiv.org/abs/1803.06959) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.04.28_SingleDirections.pdf) | 


<!--excerpt.start-->

[^1]: <sub><sup>  Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation/  Neural Machine Translation (NMT) is an end-to-end learning approach for automated translation, with the potential to overcome many of the weaknesses of conventional phrase-based translation systems. Unfortunately, NMT systems are known to be computationally expensive both in training and in translation inference. Also, most NMT systems have difficulty with rare words. These issues have hindered NMT's use in practical deployments and services, where both accuracy and speed are essential. In this work, we present GNMT, Google's Neural Machine Translation system, which attempts to address many of these issues. Our model consists of a deep LSTM network with 8 encoder and 8 decoder layers using attention and residual connections. To improve parallelism and therefore decrease training time, our attention mechanism connects the bottom layer of the decoder to the top layer of the encoder. To accelerate the final translation speed, we employ low-precision arithmetic during inference computations. To improve handling of rare words, we divide words into a limited set of common sub-word units ("wordpieces") for both input and output. This method provides a good balance between the flexibility of "character"-delimited models and the efficiency of "word"-delimited models, naturally handles translation of rare words, and ultimately improves the overall accuracy of the system. Our beam search technique employs a length-normalization procedure and uses a coverage penalty, which encourages generation of an output sentence that is most likely to cover all the words in the source sentence. On the WMT'14 English-to-French and English-to-German benchmarks, GNMT achieves competitive results to state-of-the-art. Using a human side-by-side evaluation on a set of isolated simple sentences, it reduces translation errors by an average of 60% compared to Google's phrase-based production system. </sup></sub>




[^2]: <sub><sup> Generating Sentences by Editing Prototypes, Kelvin Guu, Tatsunori B. Hashimoto, Yonatan Oren, Percy Liang / We propose a new generative model of sentences that first samples a prototype sentence from the training corpus and then edits it into a new sentence. Compared to traditional models that generate from scratch either left-to-right or by first sampling a latent sentence vector, our prototype-then-edit model improves perplexity on language modeling and generates higher quality outputs according to human evaluation. Furthermore, the model gives rise to a latent edit vector that captures interpretable semantics such as sentence similarity and sentence-level analogies. </sup></sub>
