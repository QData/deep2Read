---
toc: true
author_profile: true
sidebar:
  title: "Reviews Indexed"
  nav: sidebar-sample
layout: single
title: GNN for Program Analysis    
desc: 2019-W4
term: 2019sCourse
categories:
- 2GraphsNN
- 9DiscreteApp
tags: [ embedding, program, heterogeneous ]
---


| Presenter | Papers | Paper URL| Our Slides |
| -----: | -------------------------------------: | :----- | :----- |
| Program |   Neural network-based graph embedding for cross-platform binary code similarity detection  | [Pdf](https://openreview.net/forum?id=BJOFETxR-) + [Pdf](https://arxiv.org/abs/1708.06525)| Faizan [PDF]({{site.baseurl}}/talks2019/19sCourse/20190405-Faizan-BinaryCode.pdf)  + GaoJi [Pdf]({{site.baseurl}}/talks2019/19scribeNotes/20190405-GaoJi-Gemini.pdf) | 
| Program |  Deep Program Reidentification: A Graph Neural Network Solution | [Pdf](https://arxiv.org/abs/1812.04064) | Weilin [PDF]({{site.baseurl}}/talks2019/19sCourse/20190222-Weilin-DeepReID.pdf)  | 
| Program | Heterogeneous Graph Neural Networks for Malicious Account Detection  | [Pdf](https://dl.acm.org/citation.cfm?id=3272010) | Weilin [Pdf]({{site.baseurl}}/talks2019/19sCourse/20190315-Weilin-MaliciousAccountDetection.pdf)  | 
| Program |  Learning to represent programs with graphs | [Pdf](https://arxiv.org/abs/1812.04064) [^1]|  | 


<!--excerpt.start-->
[^1]: <sub><sup> Jack Note: Many recent works have tried NLP or CV methods to learn representations for predictive models on source code. However, these methods don't fit this data type. The main motivation here is that source code is actually a graph representation, not sequential or local. We can represent program source code as graphs and use different edge types to model syntactic and semantic relationships between different tokens of the program. To do this, we can use program’s abstract syntax tree (AST), consisting of syntax nodes and syntax tokens.  Thus, the hypothesis is that we can use use graph-based deep learning methods to learn to reason over program structures. This paper proposes to use graphs to represent both the syntactic and semantic structure of code and use graph-based deep learning methods to learn to reason over program structures. In addition, they explore how to scale Gated Graph Neural Networks training to such large graphs. We evaluate our method on two tasks: VarMisuse, in which a network attempts to predict the name of a variable given its usage, and VarNaming, in which the network learns to reason about selecting the correct variable that should be used at a given program location. Our comparison to methods that use less structured program representations shows the advantages of modeling known structure, and suggests that our models learn to infer meaningful names and to solve the VarMisuse task in many cases. Additionally, our testing showed that VarNaming identifies a number of bugs in mature open-source projects.  <sup><sub>

