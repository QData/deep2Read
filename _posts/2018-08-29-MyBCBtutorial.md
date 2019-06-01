---
layout: post
title: Survey18- My Tutorial Talk at ACM BCB18 - Interpretable Deep Learning for Genomics
desc: 2018-me
tags:
- 9MedApp
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Slides |
| -----: | ---------------------------: | :----- | :----- |
| Dr. Qi | Making Deep Learning Understandable for Analyzing Sequential Data about Gene Regulation |         |  [PDF]({{site.baseurl}}/MoreTalksTeam/20180229-deepBio-BCBtutorial.pdf) |

> #### Attend and Predict: Understanding Gene Regulation by Selective Attention on Chromatin, NIPS2017 / Ritambhara Singh, Jack Lanchantin, Arshdeep Sekhon, Yanjun Qi
>> The past decade has seen a revolution in genomic technologies that enable a flood of genome-wide profiling of chromatin marks. Recent literature tried to understand gene regulation by predicting gene expression from large-scale chromatin measurements. Two fundamental challenges exist for such learning tasks: (1) genome-wide chromatin signals are spatially structured, high-dimensional and highly modular; and (2) the core aim is to understand what are the relevant factors and how they work together? Previous studies either failed to model complex dependencies among input signals or relied on separate feature analysis to explain the decisions. This paper presents an attention-based deep learning approach; we call AttentiveChrome, that uses a unified architecture to model and to interpret dependencies among chromatin factors for controlling gene regulation. AttentiveChrome uses a hierarchy of multiple Long short-term memory (LSTM) modules to encode the input signals and to model how various chromatin marks cooperate automatically. AttentiveChrome trains two levels of attention jointly with the target prediction, enabling it to attend differentially to relevant marks and to locate important positions per mark. We evaluate the model across 56 different cell types (tasks) in human. Not only is the proposed architecture more accurate, but its attention scores also provide a better interpretation than state-of-the-art feature visualization methods such as saliency map. 
Code and data are shared at[www.deepchrome.org](http://deepchrome.org/) 