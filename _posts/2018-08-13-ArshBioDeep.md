---
layout: post
title: Application18- DNNs in a Few BioMedical Tasks
desc: 2018-team
tags:
- 3Reliable
- 6Reinforcement
- 8BioApplications
categories: 2018Reads
---


| Presenter | Papers | Information| OurPresentation |
| -----: | ----------: | :----- | :----- |
| Arshdeep |  DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning. | [PDF](https://www.ncbi.nlm.nih.gov/pubmed/30035154) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BRAIN-07242018-BlogPostsDeepLesionChestXRay.pdf) | 
| Arshdeep | Solving the RNA design problem with reinforcement learning, PLOSCB  | [PDF](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006176) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-06292018-RNAReinforcement.pdf) | 
|  Arshdeep| The CRISPR tool kit for genome editing and beyond, Mazhar Adli  | [PDF](https://www.nature.com/articles/s41467-018-04252-2) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-07062018-CRISPR-Review.pdf) | 
| Arshdeep |  deepCRISPR: optimized CRISPR guide RNA design by deep learning , Genome Biology 2018| [PDF](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1459-4) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-07162018-DeepCRISPR.pdf) | 
| Arshdeep | Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk  | [PDF](https://www.nature.com/articles/s41588-018-0160-6) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-07272018-AbinitioPrediction-SeqtoGeneExp.pdf) | 


> ####  Solving the RNA design problem with reinforcement learning, PLOSCB
>> We use reinforcement learning to train an agent for computational RNA design: given a target secondary structure, design a sequence that folds to that structure in silico. Our agent uses a novel graph convolutional architecture allowing a single model to be applied to arbitrary target structures of any length. After training it on randomly generated targets, we test it on the Eterna100 benchmark and find it outperforms all previous algorithms. Analysis of its solutions shows it has successfully learned some advanced strategies identified by players of the game Eterna, allowing it to solve some very difficult structures. On the other hand, it has failed to learn other strategies, possibly because they were not required for the targets in the training set. This suggests the possibility that future improvements to the training protocol may yield further gains in performance.
>> Author summary: Designing RNA sequences that fold to desired structures is an important problem in bioengineering. We have applied recent advances in machine learning to address this problem. The computer learns without any human input, using only trial and error to figure out how to design RNA. It quickly discovers powerful strategies that let it solve many difficult design problems. When tested on a challenging benchmark, it outperforms all previous algorithms. We analyze its solutions and identify some of the strategies it has learned, as well as other important strategies it has failed to learn. This suggests possible approaches to further improving its performance. This work reflects a paradigm shift taking place in computer science, which has the potential to transform computational biology. Instead of relying on experts to design algorithms by hand, computers can use artificial intelligence to learn their own algorithms directly. The resulting methods often work better than the ones designed by humans

> ####  deepCRISPR: optimized CRISPR guide RNA design by deep learning / Genome Biology 2018
>> A major challenge for effective application of CRISPR systems is to accurately predict the single guide RNA (sgRNA) on-target knockout efficacy and off-target profile, which would facilitate the optimized design of sgRNAs with high sensitivity and specificity. Here we present DeepCRISPR, a comprehensive computational platform to unify sgRNA on-target and off-target site prediction into one framework with deep learning, surpassing available state-of-the-art in silico tools. In addition, DeepCRISPR fully automates the identification of sequence and epigenetic features that may affect sgRNA knockout efficacy in a data-driven manner. DeepCRISPR is available at http://www.deepcrispr.net/.


> ####  Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk / Nature Geneticsvolume 50, pages1171–1179 (2018)
>>  Key challenges for human genetics, precision medicine and evolutionary biology include deciphering the regulatory code of gene expression and understanding the transcriptional effects of genome variation. However, this is extremely difficult because of the enormous scale of the noncoding mutation space. We developed a deep learning–based framework, ExPecto, that can accurately predict, ab initio from a DNA sequence, the tissue-specific transcriptional effects of mutations, including those that are rare or that have not been observed. We prioritized causal variants within disease- or trait-associated loci from all publicly available genome-wide association studies and experimentally validated predictions for four immune-related diseases. By exploiting the scalability of ExPecto, we characterized the regulatory mutation space for human RNA polymerase II–transcribed genes by in silico saturation mutagenesis and profiled > 140 million promoter-proximal mutations. This enables probing of evolutionary constraints on gene expression and ab initio prediction of mutation disease effects, making ExPecto an end-to-end computational framework for the in silico prediction of expression and disease risk.