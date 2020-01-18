---
layout: post
title: Basic16- Basic Deep NN and Robustness 
desc: 2017-team
categories:
- 0Basics
- 3Reliable
term: 2017Reads
tags: [  Adversarial-Examples, robustness, visualizing, Interpretable, Certified-Defense ]
---



| Presenter | Papers | Paper URL| Our Slides |
| -----: | -------------------------------------: | :----- | :----- |
<!--header-->
| AE |Intriguing properties of neural networks /  | [PDF](https://arxiv.org/abs/1312.6199) |  |
| AE | Explaining and Harnessing Adversarial Examples | [PDF](https://arxiv.org/abs/1412.6572) |  |
| AE | Towards Deep Learning Models Resistant to Adversarial Attacks | [PDF](https://arxiv.org/abs/1706.06083) | | 
| AE | DeepFool: a simple and accurate method to fool deep neural networks | [PDF](https://arxiv.org/abs/1511.04599) | | 
| AE | Towards Evaluating the Robustness of Neural Networks by Carlini and Wagner | [PDF](https://arxiv.org/abs/1608.04644) | [PDF]({{site.baseurl}}/MoreTalksTeam/Jack/20170512_towards_evaluating_the_robustness_of_neural_networks.pdf)| 
| Data | Basic Survey of ImageNet - LSVRC competition | [URL](http://www.image-net.org/) | [PDF]({{site.baseurl}}/MoreTalksTeam/Jack/20160722ImageNet-LSVRC-2010-2015.pdf) | 
| Understand | Understanding Black-box Predictions via Influence Functions | [PDF](https://arxiv.org/abs/1703.04730) |  |
| Understand | Deep inside convolutional networks: Visualising image classification models and saliency maps | [PDF](https://arxiv.org/abs/1312.6034) |  |
| Understand | BeenKim, Interpretable Machine Learning, ICML17 Tutorial [^1]| [PDF](https://people.csail.mit.edu/beenkim/papers/BeenK_FinaleDV_ICML2017_tutorial.pdf) |  |
| provable | Provable defenses against adversarial examples via the convex outer adversarial polytope, Eric Wong, J. Zico Kolter, | [URL](https://arxiv.org/abs/1711.00851) | | 

<!--excerpt.good-->

[^1] Notes about Interpretable Machine Learning

#### Notes of Interpretability in Machine Learning from Been Kim Tutorial 
by Brandon Liu

###### Important Criteria in ML Systems
+ Safety
+ Nondiscrimination
+ Avoiding technical debt
+ Providing the right to explanation
+ Ex. Self driving cars and other autonomous vehicles - almost impossible to come up with all possible unit tests.

###### What is interpretability?
+ The ability to give explanations to humans. 

###### Two Branches of Interpretability
+ In the context of an application:	if the system is useful in either a practical application or a simplified version of it, then it must be somehow interpretable.
+ Via a quantifiable proxy: a researcher might first claim that some model class—e.g. sparse linear models, rule lists, gradient boosted trees—are interpretable and then present algorithms to optimize within that class.

###### Before building any model	
+ Visualization
+ Exploratory data analysis


###### Building a new model	
+ Rule-based, per-feature-based
+ Case-based
+ Sparsity
+ Monotonicity

######  After building a model
+ Sensitivity analysis, gradient-based methods
+ mimic/surrogate models
+ Investigation on hidden layers
