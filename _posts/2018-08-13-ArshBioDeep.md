---
layout: post
title: Application18- DNNs in a Few BioMedical Tasks
desc: 2018-team
tags:
- 3Reliable
- 6Reinforcement
- 8MedApplications
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Arshdeep |  DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning.  | [PDF](https://www.ncbi.nlm.nih.gov/pubmed/30035154) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BRAIN-07242018-BlogPostsDeepLesionChestXRay.pdf) | 
| Arshdeep | Solving the RNA design problem with reinforcement learning, PLOSCB  [^1] | [PDF](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006176) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-06292018-RNAReinforcement.pdf) | 
| Arshdeep | Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk [^2] | [PDF](https://www.nature.com/articles/s41588-018-0160-6) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-07272018-AbinitioPrediction-SeqtoGeneExp.pdf) | 
| Arshdeep | Towards Gene Expression Convolutions using Gene Interaction Graphs, Francis Dutil, Joseph Paul Cohen, Martin Weiss, Georgy Derevyanko, Yoshua Bengio [^3] | [PDF](https://arxiv.org/abs/1806.06975) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-06292018-GeneInterCNN.pdf) | 
|  Brandon| Kipoi: Accelerating the Community Exchange and Reuse of Predictive Models for Genomics | [PDF](http://kipoi.org/docs/) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Brandon18.9.14Kipoi.pdf) | 






[^3]: <sub><sup> Towards Gene Expression Convolutions using Gene Interaction Graphs, Francis Dutil, Joseph Paul Cohen, Martin Weiss, Georgy Derevyanko, Yoshua Bengio/ We study the challenges of applying deep learning to gene expression data. We find experimentally that there exists non-linear signal in the data, however is it not discovered automatically given the noise and low numbers of samples used in most research. We discuss how gene interaction graphs (same pathway, protein-protein, co-expression, or research paper text association) can be used to impose a bias on a deep model similar to the spatial bias imposed by convolutions on an image. We explore the usage of Graph Convolutional Neural Networks coupled with dropout and gene embeddings to utilize the graph information. We find this approach provides an advantage for particular tasks in a low data regime but is very dependent on the quality of the graph used. We conclude that more work should be done in this direction. We design experiments that show why existing methods fail to capture signal that is present in the data when features are added which clearly isolates the problem that needs to be addressed. </sup></sub>






[^1]: <sub><sup> Solving the RNA design problem with reinforcement learning, PLOSCB/ We use reinforcement learning to train an agent for computational RNA design: given a target secondary structure, design a sequence that folds to that structure in silico. Our agent uses a novel graph convolutional architecture allowing a single model to be applied to arbitrary target structures of any length. After training it on randomly generated targets, we test it on the Eterna100 benchmark and find it outperforms all previous algorithms. Analysis of its solutions shows it has successfully learned some advanced strategies identified by players of the game Eterna, allowing it to solve some very difficult structures. On the other hand, it has failed to learn other strategies, possibly because they were not required for the targets in the training set. This suggests the possibility that future improvements to the training protocol may yield further gains in performance. Author summary: Designing RNA sequences that fold to desired structures is an important problem in bioengineering. We have applied recent advances in machine learning to address this problem. The computer learns without any human input, using only trial and error to figure out how to design RNA. It quickly discovers powerful strategies that let it solve many difficult design problems. When tested on a challenging benchmark, it outperforms all previous algorithms. We analyze its solutions and identify some of the strategies it has learned, as well as other important strategies it has failed to learn. This suggests possible approaches to further improving its performance. This work reflects a paradigm shift taking place in computer science, which has the potential to transform computational biology. Instead of relying on experts to design algorithms by hand, computers can use artificial intelligence to learn their own algorithms directly. The resulting methods often work better than the ones designed by humans. </sup></sub>



[^2]: <sub><sup>  Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk / Nature Geneticsvolume 50, pages1171–1179 (2018)/ Key challenges for human genetics, precision medicine and evolutionary biology include deciphering the regulatory code of gene expression and understanding the transcriptional effects of genome variation. However, this is extremely difficult because of the enormous scale of the noncoding mutation space. We developed a deep learning–based framework, ExPecto, that can accurately predict, ab initio from a DNA sequence, the tissue-specific transcriptional effects of mutations, including those that are rare or that have not been observed. We prioritized causal variants within disease- or trait-associated loci from all publicly available genome-wide association studies and experimentally validated predictions for four immune-related diseases. By exploiting the scalability of ExPecto, we characterized the regulatory mutation space for human RNA polymerase II–transcribed genes by in silico saturation mutagenesis and profiled > 140 million promoter-proximal mutations. This enables probing of evolutionary constraints on gene expression and ab initio prediction of mutation disease effects, making ExPecto an end-to-end computational framework for the in silico prediction of expression and disease risk. </sup></sub>