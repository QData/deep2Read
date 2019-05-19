---
layout: post
title:  
tags:
- About
---

## About this website:

+ As a group, we need to improve our knowledge of the fast-growing field of deep learning
+ To educate new members with basic tutorials, and to help existing members understand advanced topics.
+ This website includes a (growing) list of tutorials and papers we survey for such a purpose.
+ We hope this website is helpful to people who share similar research interests or are interested with learning advanced topics about deep learning.
+ Please feel free to email me (yanjun@virginia.edu), if you have related comments, questions or recommendations.

## Claim 
+ BTW: The covered tutorials and papers are by no means an exhaustive list, but are topics which we have learned or plan to learn in our reading group.

## History

+ This website was started from the seminar course "Advances in Deep Learning" I taught at UVA in Fall 2017. Later I expand the content with my team reaing group and another course I taught in Spring 2019. 

+ The materials aim to offer opportunities for students to have in-depth understanding and hands-on experience of advances in deep learning. 

<hr>
--- ---
layout: post
title: Structures17- Memory-Augmented Networks 
desc: 2017-team
tags:
- 2Structures
categories: 2017Reads
---

| Presenter | Papers | Paper URL| OurPresentation |
| -----: | ---------------------------: | :----- | :----- |
| Muthu | Matching Networks for One Shot Learning (NIPS16) [^1]| [PDF](https://arxiv.org/abs/1606.04080) | [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-MatchingNet.pdf) |
| Jack | Meta-Learning with Memory-Augmented Neural Networks (ICML16) [^2]| [PDF](http://proceedings.mlr.press/v48/santoro16.pdf) | [PDF]({{site.baseurl}}/MoreTalksTeam/Jack/20170724_Memory_Augmented.pdf) |


[^1]: <sub><sup> Matching Networks for One Shot Learning (NIPS16): Learning from a few examples remains a key challenge in machine learning. Despite recent advances in important domains such as vision and language, the standard supervised deep learning paradigm does not offer a satisfactory solution for learning new concepts rapidly from little data. In this work, we employ ideas from metric learning based on deep neural features and from recent advances that augment neural networks with external memories. Our framework learns a network that maps a small labelled support set and an unlabelled example to its label, obviating the need for fine-tuning to adapt to new class types. We then define one-shot learning problems on vision (using Omniglot, ImageNet) and language tasks. Our algorithm improves one-shot accuracy on ImageNet from 87.6% to 93.2% and from 88.0% to 93.8% on Omniglot compared to competing approaches. We also demonstrate the usefulness of the same model on language modeling by introducing a one-shot task on the Penn Treebank. </sup></sub>


[^2]: <sub><sup> Meta-Learning with Memory-Augmented Neural Networks (ICML16) Despite recent breakthroughs in the applications of deep neural networks, one setting that presents a persistent challenge is that of "one-shot learning." Traditional gradient-based networks require a lot of data to learn, often through extensive iterative training. When new data is encountered, the models must inefficiently relearn their parameters to adequately incorporate the new information without catastrophic interference. Architectures with augmented memory capacities, such as Neural Turing Machines (NTMs), offer the ability to quickly encode and retrieve new information, and hence can potentially obviate the downsides of conventional models. Here, we demonstrate the ability of a memory-augmented neural network to rapidly assimilate new data, and leverage this data to make accurate predictions after only a few samples. We also introduce a new method for accessing an external memory that focuses on memory content, unlike previous methods that additionally use memory location-based focusing mechanisms. </sup></sub>

---
layout: post
title: Reliable17-Secure Machine Learning
desc: 2017-team
categories: 2017Reads
tags:
- 3Reliable
---



| Presenter | Papers | Paper URL| OurPresentation |
| -----: | ---------------------------: | :----- | :----- |
| Tobin | Summary of A few Papers on: Machine Learning and Cryptography, (e.g., learning to Protect Communications with Adversarial Neural Cryptography) [^1] | [PDF](https://arxiv.org/abs/1610.06918) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Tobin-CryptoML.pdf) |
| Tobin |  Privacy Aware Learning (NIPS12) [^2]| [PDF](https://web.stanford.edu/~jduchi/projects/DuchiJoWa12_nips.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Tobin-DuchiPrivacyLearning.pdf) | 
| Tobin |  Can Machine Learning be Secure?(2006) | [PDF](http://bnrg.cs.berkeley.edu/~adj/publications/paper-files/asiaccs06.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Tobin-SecureLearning.pdf)  | 


[^1]: <sub><sup> Learning to protect communications with adversarial neural cryptography Abadi & Anderson, arXiv 2016:  We ask whether neural networks can learn to use secret keys to protect information from other neural networks.  Specifically, we focus on ensuring confidentiality properties in a multiagent system, and we specify those properties in terms of an adversary.  Thus, a system may consist of neural networks named Alice and Bob, and we aim to limit what a third neural network named Eve learns from eavesdropping on the communication between Alice and Bob. We do not prescribe specific cryptographic algorithms to these neural networks; instead, we train end-to-end, adversarially. We demonstrate that the neural networks can learn how to perform forms of encryption and decryption, and also how to apply these operations selectively in order to meet confidentiality goals. </sup></sub>



[^2]: <sub><sup> Privacy Aware Learning (NIPS12) / John C. Duchi, Michael I. Jordan, Martin J. Wainwright/  We study statistical risk minimization problems under a privacy model in which the data is kept confidential even from the learner. In this local privacy framework, we establish sharp upper and lower bounds on the convergence rates of statistical estimation procedures. As a consequence, we exhibit a precise tradeoff between the amount of privacy the data preserves and the utility, as measured by convergence rate, of any statistical estimator or learning procedure. </sup></sub>


 
---
layout: post
title: Structures17 -Adaptive Deep Networks I
desc: 2017-team
tags:
- 2Structures
categories: 2017Reads
---



| Presenter | Papers | Paper URL| OurPresentation |
| -----: | ---------------------------: | :----- | :----- |
| Arshdeep |  HyperNetworks, David Ha, Andrew Dai, Quoc V. Le ICLR 2017 [^1] | [PDF](https://arxiv.org/abs/1609.09106)  |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/17-01-hypernetworks_parameter_prediction.pdf) | 
| Arshdeep |  Learning feed-forward one-shot learners [^2]| [PDF](https://arxiv.org/abs/1606.05233) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/17-02-one_shot_parameter_prediction.pdf) | 
| Arshdeep |  Learning to Learn by gradient descent by gradient descent [^3]| [PDF](https://arxiv.org/abs/1606.04474) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/17-03-gradient_descent_parameter_prediction.pdf) | 
|  Arshdeep | Dynamic Filter Networks [^4] https://arxiv.org/abs/1605.09673 | [PDF](https://arxiv.org/abs/1605.09673) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/17-04-dynamic-filter-networks.pdf) | 


[^1]: <sub><sup> HyperNetworks, David Ha, Andrew Dai, Quoc V. Le ICLR 2017 / This work explores hypernetworks: an approach of using a one network, also known as a hypernetwork, to generate the weights for another network. Hypernetworks provide an abstraction that is similar to what is found in nature: the relationship between a genotype - the hypernetwork - and a phenotype - the main network. Though they are also reminiscent of HyperNEAT in evolution, our hypernetworks are trained end-to-end with backpropagation and thus are usually faster. The focus of this work is to make hypernetworks useful for deep convolutional networks and long recurrent networks, where hypernetworks can be viewed as relaxed form of weight-sharing across layers. Our main result is that hypernetworks can generate non-shared weights for LSTM and achieve near state-of-the-art results on a variety of sequence modelling tasks including character-level language modelling, handwriting generation and neural machine translation, challenging the weight-sharing paradigm for recurrent networks. Our results also show that hypernetworks applied to convolutional networks still achieve respectable results for image recognition tasks compared to state-of-the-art baseline models while requiring fewer learnable parameters. </sup></sub>

[^2]: <sub><sup> Learning feed-forward one-shot learners, Arxiv  2016, Luca Bertinetto, João F. Henriques, Jack Valmadre, Philip H. S. Torr, Andrea Vedaldi/ One-shot learning is usually tackled by using generative models or discriminative embeddings. Discriminative methods based on deep learning, which are very effective in other learning scenarios, are ill-suited for one-shot learning as they need large amounts of training data. In this paper, we propose a method to learn the parameters of a deep model in one shot. We construct the learner as a second deep network, called a learnet, which predicts the parameters of a pupil network from a single exemplar. In this manner we obtain an efficient feed-forward one-shot learner, trained end-to-end by minimizing a one-shot classification objective in a learning to learn formulation. In order to make the construction feasible, we propose a number of factorizations of the parameters of the pupil network. We demonstrate encouraging results by learning characters from single exemplars in Omniglot, and by tracking visual objects from a single initial exemplar in the Visual Object Tracking benchmark. </sup></sub>


[^3]: <sub><sup> Learning to Learn by gradient descent by gradient descent, Arxiv  2016/ The move from hand-designed features to learned features in machine learning has been wildly successful. In spite of this, optimization algorithms are still designed by hand. In this paper we show how the design of an optimization algorithm can be cast as a learning problem, allowing the algorithm to learn to exploit structure in the problems of interest in an automatic way. Our learned algorithms, implemented by LSTMs, outperform generic, hand-designed competitors on the tasks for which they are trained, and also generalize well to new tasks with similar structure. We demonstrate this on a number of tasks, including simple convex problems, training neural networks, and styling images with neural art. </sup></sub>



[^4]: <sub><sup> Dynamic Filter Networks, Bert De Brabandere, Xu Jia, Tinne Tuytelaars, Luc Van Gool (Submitted on 31 May 2016 (v1), last revised 6 Jun 2016 (this version, v2))/ In a traditional convolutional layer, the learned filters stay fixed after training. In contrast, we introduce a new framework, the Dynamic Filter Network, where filters are generated dynamically conditioned on an input. We show that this architecture is a powerful one, with increased flexibility thanks to its adaptive nature, yet without an excessive increase in the number of model parameters. A wide variety of filtering operations can be learned this way, including local spatial transformations, but also others like selective (de)blurring or adaptive feature extraction. Moreover, multiple such layers can be combined, e.g. in a recurrent architecture. We demonstrate the effectiveness of the dynamic filter network on the tasks of video and stereo prediction, and reach state-of-the-art performance on the moving MNIST dataset with a much smaller model. By visualizing the learned filters, we illustrate that the network has picked up flow information by only looking at unlabelled training data. This suggests that the network can be used to pretrain networks for various supervised tasks in an unsupervised way, like optical flow and depth estimation. </sup></sub>
---
layout: post
title: Structures17- DNN based Embedding 
desc: 2017-team
tags:
- 2Structures
categories: 2017Reads
---

| Presenter | Papers | Paper URL| OurPresentation |
| -----: | ---------------------------: | :----- | :----- |
| Muthu | NeuroIPS Embedding Papers survey 2012 to 2015| [NIPS](https://papers.nips.cc/) | [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-NIPSEmbedding12to15.pdf) |
| Tobin | Binary embeddings with structured hashed projections [^1] | [PDF](https://arxiv.org/abs/1511.05212) | [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Tobin-BinaryEmbedding.pdf) |


[^1]: <sub><sup> Binary embeddings with structured hashed projections /Anna Choromanska, Krzysztof Choromanski, Mariusz Bojarski, Tony Jebara, Sanjiv Kumar, Yann LeCun (Submitted on 16 Nov 2015 (v1), last revised 1 Jul 2016 (this version, v5))/ We consider the hashing mechanism for constructing binary embeddings, that involves pseudo-random projections followed by nonlinear (sign function) mappings. The pseudo-random projection is described by a matrix, where not all entries are independent random variables but instead a fixed "budget of randomness" is distributed across the matrix. Such matrices can be efficiently stored in sub-quadratic or even linear space, provide reduction in randomness usage (i.e. number of required random values), and very often lead to computational speed ups. We prove several theoretical results showing that projections via various structured matrices followed by nonlinear mappings accurately preserve the angular distance between input high-dimensional vectors. To the best of our knowledge, these results are the first that give theoretical ground for the use of general structured matrices in the nonlinear setting. In particular, they generalize previous extensions of the Johnson-Lindenstrauss lemma and prove the plausibility of the approach that was so far only heuristically confirmed for some special structured matrices. Consequently, we show that many structured matrices can be used as an efficient information compression mechanism. Our findings build a better understanding of certain deep architectures, which contain randomly weighted and untrained layers, and yet achieve high performance on different learning tasks. We empirically verify our theoretical findings and show the dependence of learning via structured hashed projections on the performance of neural network as well as nearest neighbor classifier. </sup></sub>---
layout: post
title: Optimization17- Optimization in DNN
desc: 2017-team
categories: 2017Reads
tags:
- 4Optimization
---


| Presenter | Papers | Paper URL| OurPresentation |
| -----: | ---------------------------: | :----- | :----- |
| Muthu | Optimization Methods for Large-Scale Machine Learning, Léon Bottou, Frank E. Curtis, Jorge Nocedal [^1] | [PDF](https://arxiv.org/abs/1606.04838) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-OptmOptimizationSurvey.pdf) | 
| Muthu | Fast Training of Recurrent Networks Based on EM Algorithm (1998) [^2]  | [PDF](https://pdfs.semanticscholar.org/f64f/4fdfbf4a7658763c96d48efead811b3683ab.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-OptmAlternatives.pdf) | 
| Muthu |  FitNets: Hints for Thin Deep Nets, ICLR15  [^3]| [PDF](https://arxiv.org/abs/1412.6550) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-OptmBengio.pdf) | 
| Muthu | Two NIPS 2015 Deep Learning Optimization Papers  | [PDF]() |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-OptmNIPS15.pdf) | 
| Muthu |  Difference Target Propagation (2015) [^4]| [PDF](https://arxiv.org/abs/1412.7525) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-OptmTarget.pdf) | 


[^1]: <sub><sup> Optimization Methods for Large-Scale Machine Learning, Léon Bottou, Frank E. Curtis, Jorge Nocedal / This paper provides a review and commentary on the past, present, and future of numerical optimization algorithms in the context of machine learning applications. Through case studies on text classification and the training of deep neural networks, we discuss how optimization problems arise in machine learning and what makes them challenging. A major theme of our study is that large-scale machine learning represents a distinctive setting in which the stochastic gradient (SG) method has traditionally played a central role while conventional gradient-based nonlinear optimization techniques typically falter. Based on this viewpoint, we present a comprehensive theory of a straightforward, yet versatile SG algorithm, discuss its practical behavior, and highlight opportunities for designing algorithms with improved performance. This leads to a discussion about the next generation of optimization methods for large-scale machine learning, including an investigation of two main streams of research on techniques that diminish noise in the stochastic directions and methods that make use of second-order derivative approximations. </sup></sub>


[^2]: <sub><sup> Fast Training of Recurrent Networks Based on EM Algorithm (1998)  / In this work, a probabilistic model is established for recurrent networks. The expectation-maximization (EM) algorithm is then applied to derive a new fast training algorithm for recurrent networks through mean-field approximation. This new algorithm converts training a complicated recurrent network into training an array of individual feedforward neurons. These neurons are then trained via a linear weighted regression algorithm. The training time has been improved by five to 15 times on benchmark problems. Published in: IEEE Transactions on Neural Networks ( Volume: 9 , Issue: 1 , Jan 1998 )  </sup></sub>


[^3]: <sub><sup> FitNets: Hints for Thin Deep Nets, ICLR15 / Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio (Submitted on 19 Dec 2014 (v1), last revised 27 Mar 2015 (this version, v4)) While depth tends to improve network performances, it also makes gradient-based training more difficult since deeper networks tend to be more non-linear. The recently proposed knowledge distillation approach is aimed at obtaining small and fast-to-execute models, and it has shown that a student network could imitate the soft output of a larger teacher network or ensemble of networks. In this paper, we extend this idea to allow the training of a student that is deeper and thinner than the teacher, using not only the outputs but also the intermediate representations learned by the teacher as hints to improve the training process and final performance of the student. Because the student intermediate hidden layer will generally be smaller than the teacher's intermediate hidden layer, additional parameters are introduced to map the student hidden layer to the prediction of the teacher hidden layer. This allows one to train deeper students that can generalize better or run faster, a trade-off that is controlled by the chosen student capacity. For example, on CIFAR-10, a deep student network with almost 10.4 times less parameters outperforms a larger, state-of-the-art teacher network.  </sup></sub>


[^4]: <sub><sup> Difference Target Propagation (2015) / 	13 pages, 8 figures, Accepted in ECML/PKDD 2015 / Dong-Hyun Lee, Saizheng Zhang, Asja Fischer, Yoshua Bengio/ Back-propagation has been the workhorse of recent successes of deep learning but it relies on infinitesimal effects (partial derivatives) in order to perform credit assignment. This could become a serious issue as one considers deeper and more non-linear functions, e.g., consider the extreme case of nonlinearity where the relation between parameters and cost is actually discrete. Inspired by the biological implausibility of back-propagation, a few approaches have been proposed in the past that could play a similar credit assignment role. In this spirit, we explore a novel approach to credit assignment in deep networks that we call target propagation. The main idea is to compute targets rather than gradients, at each layer. Like gradients, they are propagated backwards. In a way that is related but different from previously proposed proxies for back-propagation which rely on a backwards network with symmetric weights, target propagation relies on auto-encoders at each layer. Unlike back-propagation, it can be applied even when units exchange stochastic bits rather than real numbers. We show that a linear correction for the imperfectness of the auto-encoders, called difference target propagation, is very effective to make target propagation actually work, leading to results comparable to back-propagation for deep networks with discrete and continuous units and denoising auto-encoders and achieving state of the art for stochastic networks. </sup></sub>---
layout: post
title: Generative17- Generative Deep Networks 
desc: 2017-team
tags:
- 5Generative
categories: 2017Reads
---


| Presenter | Papers | Paper URL| OurPresentation |
| -----: | ---------------------------: | :----- | :----- |
| Tobin |  Energy-Based Generative Adversarial Network [^1] | [PDF](https://arxiv.org/abs/1609.03126) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Tobin-EnergyGAN.pdf) | 
| Jack |  ThreeDeepGenerativeModels |  [PDF]() |  [PDF]({{site.baseurl}}/MoreTalksTeam/Jack/04_08_16-JackThreeDeepGenerativeModels.pdf) | 
| Muthu |  Deep Compression: Compressing Deep Neural Networks (ICLR 2016) [^2]| [PDF](https://arxiv.org/abs/1510.00149) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un17/Muthu-Compression.pdf) | 


[^1]: <sub><sup> Energy-Based Generative Adversarial Network, Junbo Zhao, Michael Mathieu, Yann LeCun (Submitted on 11 Sep 2016 (v1), last revised 6 Mar 2017 (this version, v4))/ We introduce the "Energy-based Generative Adversarial Network" model (EBGAN) which views the discriminator as an energy function that attributes low energies to the regions near the data manifold and higher energies to other regions. Similar to the probabilistic GANs, a generator is seen as being trained to produce contrastive samples with minimal energies, while the discriminator is trained to assign high energies to these generated samples. Viewing the discriminator as an energy function allows to use a wide variety of architectures and loss functionals in addition to the usual binary classifier with logistic output. Among them, we show one instantiation of EBGAN framework as using an auto-encoder architecture, with the energy being the reconstruction error, in place of the discriminator. We show that this form of EBGAN exhibits more stable behavior than regular GANs during training. We also show that a single-scale architecture can be trained to generate high-resolution images. </sup></sub>


[^2]: <sub><sup> Deep Compression: Compressing Deep Neural Networks (ICLR 2016) / Song Han, Huizi Mao, William J. Dally / conference paper at ICLR 2016 (oral) / Neural networks are both computationally intensive and memory intensive, making them difficult to deploy on embedded systems with limited hardware resources. To address this limitation, we introduce "deep compression", a three stage pipeline: pruning, trained quantization and Huffman coding, that work together to reduce the storage requirement of neural networks by 35x to 49x without affecting their accuracy. Our method first prunes the network by learning only the important connections. Next, we quantize the weights to enforce weight sharing, finally, we apply Huffman coding. After the first two steps we retrain the network to fine tune the remaining connections and the quantized centroids. Pruning, reduces the number of connections by 9x to 13x; Quantization then reduces the number of bits that represent each connection from 32 to 5. On the ImageNet dataset, our method reduced the storage required by AlexNet by 35x, from 240MB to 6.9MB, without loss of accuracy. Our method reduced the size of VGG-16 by 49x from 552MB to 11.3MB, again with no loss of accuracy. This allows fitting the model into on-chip SRAM cache rather than off-chip DRAM memory. Our compression method also facilitates the use of complex neural networks in mobile applications where application size and download bandwidth are constrained. Benchmarked on CPU, GPU and mobile GPU, compressed network has 3x to 4x layerwise speedup and 3x to 7x better energy efficiency. </sup></sub>---
layout: post
title: Structures17 - Adaptive Deep Networks II
desc: 2017-team
tags:
- 2Structures
categories: 2017Reads
---


| Presenter | Papers | Paper URL| OurPresentation |
| -----: | ---------------------------: | :----- | :----- |
| Arshdeep | Image Question Answering using Convolutional Neural Network with Dynamic Parameter Prediction [^1] | [PDF](https://arxiv.org/abs/1511.05756) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/17-05-vqa-cnn.pdf) | 
| Arshdeep |  Decoupled Neural Interfaces Using Synthetic Gradients [^2]| [PDF](https://arxiv.org/abs/1608.05343) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/17-06-synthetic-gradients.pdf) | 
| Arshdeep |  Diet Networks: Thin Parameters for Fat Genomics [^3] | [PDF](https://arxiv.org/abs/1611.09340) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/17-07dietnet.pdf) | 
| Arshdeep |  Metric Learning with Adaptive Density Discrimination [^4]| [PDF](https://arxiv.org/abs/1511.05939) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/17-08-DistanceMetricLearning.pdf) | 


[^1]: <sub><sup> Image Question Answering using Convolutional Neural Network with Dynamic Parameter Prediction / Hyeonwoo Noh, Paul Hongsuck Seo, Bohyung Han (Submitted on 18 Nov 2015)/ We tackle image question answering (ImageQA) problem by learning a convolutional neural network (CNN) with a dynamic parameter layer whose weights are determined adaptively based on questions. For the adaptive parameter prediction, we employ a separate parameter prediction network, which consists of gated recurrent unit (GRU) taking a question as its input and a fully-connected layer generating a set of candidate weights as its output. However, it is challenging to construct a parameter prediction network for a large number of parameters in the fully-connected dynamic parameter layer of the CNN. We reduce the complexity of this problem by incorporating a hashing technique, where the candidate weights given by the parameter prediction network are selected using a predefined hash function to determine individual weights in the dynamic parameter layer. The proposed network---joint network with the CNN for ImageQA and the parameter prediction network---is trained end-to-end through back-propagation, where its weights are initialized using a pre-trained CNN and GRU. The proposed algorithm illustrates the state-of-the-art performance on all available public ImageQA benchmarks. </sup></sub>


[^2]: <sub><sup> Decoupled Neural Interfaces Using Synthetic Gradients / Max Jaderberg, Wojciech Marian Czarnecki, Simon Osindero, Oriol Vinyals, Alex Graves, David Silver, Koray Kavukcuoglu (Submitted on 18 Aug 2016 (v1), last revised 3 Jul 2017 (this version, v2))/ Training directed neural networks typically requires forward-propagating data through a computation graph, followed by backpropagating error signal, to produce weight updates. All layers, or more generally, modules, of the network are therefore locked, in the sense that they must wait for the remainder of the network to execute forwards and propagate error backwards before they can be updated. In this work we break this constraint by decoupling modules by introducing a model of the future computation of the network graph. These models predict what the result of the modelled subgraph will produce using only local information. In particular we focus on modelling error gradients: by using the modelled synthetic gradient in place of true backpropagated error gradients we decouple subgraphs, and can update them independently and asynchronously i.e. we realise decoupled neural interfaces. We show results for feed-forward models, where every layer is trained asynchronously, recurrent neural networks (RNNs) where predicting one's future gradient extends the time over which the RNN can effectively model, and also a hierarchical RNN system with ticking at different timescales. Finally, we demonstrate that in addition to predicting gradients, the same framework can be used to predict inputs, resulting in models which are decoupled in both the forward and backwards pass -- amounting to independent networks which co-learn such that they can be composed into a single functioning corporation. </sup></sub>


[^3]: <sub><sup> Diet Networks: Thin Parameters for Fat Genomics / Adriana Romero, Pierre Luc Carrier, Akram Erraqabi, Tristan Sylvain, Alex Auvolat, Etienne Dejoie, Marc-André Legault, Marie-Pierre Dubé, Julie G. Hussin, Yoshua Bengio / ICLR17/ Learning tasks such as those involving genomic data often poses a serious challenge: the number of input features can be orders of magnitude larger than the number of training examples, making it difficult to avoid overfitting, even when using the known regularization techniques. We focus here on tasks in which the input is a description of the genetic variation specific to a patient, the single nucleotide polymorphisms (SNPs), yielding millions of ternary inputs. Improving the ability of deep learning to handle such datasets could have an important impact in precision medicine, where high-dimensional data regarding a particular patient is used to make predictions of interest. Even though the amount of data for such tasks is increasing, this mismatch between the number of examples and the number of inputs remains a concern. Naive implementations of classifier neural networks involve a huge number of free parameters in their first layer: each input feature is associated with as many parameters as there are hidden units. We propose a novel neural network parametrization which considerably reduces the number of free parameters. It is based on the idea that we can first learn or provide a distributed representation for each input feature (e.g. for each position in the genome where variations are observed), and then learn (with another neural network called the parameter prediction network) how to map a feature's distributed representation to the vector of parameters specific to that feature in the classifier neural network (the weights which link the value of the feature to each of the hidden units). We show experimentally on a population stratification task of interest to medical studies that the proposed approach can significantly reduce both the number of parameters and the error rate of the classifier. </sup></sub>


[^4]: <sub><sup> Metric Learning with Adaptive Density Discrimination / ICLR 2016 / Oren Rippel, Manohar Paluri, Piotr Dollar, Lubomir Bourdev/ Distance metric learning (DML) approaches learn a transformation to a representation space where distance is in correspondence with a predefined notion of similarity. While such models offer a number of compelling benefits, it has been difficult for these to compete with modern classification algorithms in performance and even in feature extraction. In this work, we propose a novel approach explicitly designed to address a number of subtle yet important issues which have stymied earlier DML algorithms. It maintains an explicit model of the distributions of the different classes in representation space. It then employs this knowledge to adaptively assess similarity, and achieve local discrimination by penalizing class distribution overlap. We demonstrate the effectiveness of this idea on several tasks. Our approach achieves state-of-the-art classification results on a number of fine-grained visual recognition datasets, surpassing the standard softmax classifier and outperforming triplet loss by a relative margin of 30-40%. In terms of computational performance, it alleviates training inefficiencies in the traditional triplet loss, reaching the same error in 5-30 times fewer iterations. Beyond classification, we further validate the saliency of the learnt representations via their attribute concentration and hierarchy recovery properties, achieving 10-25% relative gains on the softmax classifier and 25-50% on triplet loss in these tasks. </sup></sub>
---
layout: post
title: Reliable17-Testing and Machine Learning Basics 
desc: 2017-team
tags:
- 3Reliable
categories: 2017Reads
---


| Presenter | Papers | Paper URL| OurPresentation |
| -----: | ---------------------------: | :----- | :----- |
| GaoJi |  A few useful things to know about machine learning | [PDF](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/17Ji_MachineLearning.pdf) | 
| GaoJi | A few papers related to testing learning, e.g., Understanding Black-box Predictions via Influence Functions  | [PDF](https://arxiv.org/abs/1703.04730) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/17Ji_AfewPapersTestML.pdf) | 
| GaoJi |  Automated White-box Testing of Deep Learning Systems [^1] | [PDF](http://www.cs.columbia.edu/~junfeng/papers/deepxplore-sosp17.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/17Ji_DeepXplore.pdf) | 
| GaoJi |   Testing and Validating Machine Learning Classifiers by Metamorphic Testing [^2] | [PDF](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3082144/) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/17Ji-MetaTestML.pdf) | 
| GaoJi | Software testing: a research travelogue (2000–2014)  | [PDF](https://dl.acm.org/citation.cfm?id=2593885) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/17Ji-TestMLTravelogue.pdf) | 


[^1]: <sub><sup> DeepXplore: Automated Whitebox Testing of Deep Learning Systems / Kexin Pei, Yinzhi Cao, Junfeng Yang, Suman Jana / published in SOSP'17/ Deep learning (DL) systems are increasingly deployed in safety- and security-critical domains including self-driving cars and malware detection, where the correctness and predictability of a system's behavior for corner case inputs are of great importance. Existing DL testing depends heavily on manually labeled data and therefore often fails to expose erroneous behaviors for rare inputs. We design, implement, and evaluate DeepXplore, the first whitebox framework for systematically testing real-world DL systems. First, we introduce neuron coverage for systematically measuring the parts of a DL system exercised by test inputs. Next, we leverage multiple DL systems with similar functionality as cross-referencing oracles to avoid manual checking. Finally, we demonstrate how finding inputs for DL systems that both trigger many differential behaviors and achieve high neuron coverage can be represented as a joint optimization problem and solved efficiently using gradient-based search techniques.  DeepXplore efficiently finds thousands of incorrect corner case behaviors (e.g., self-driving cars crashing into guard rails and malware masquerading as benign software) in state-of-the-art DL models with thousands of neurons trained on five popular datasets including ImageNet and Udacity self-driving challenge data. For all tested DL models, on average, DeepXplore generated one test input demonstrating incorrect behavior within one second while running only on a commodity laptop. We further show that the test inputs generated by DeepXplore can also be used to retrain the corresponding DL model to improve the model's accuracy by up to 3%. </sup></sub>


[^2]: <sub><sup> Testing and Validating Machine Learning Classifiers by Metamorphic Testing / 2011/ Abstract: Machine learning algorithms have provided core functionality to many application domains - such as bioinformatics, computational linguistics, etc. However, it is difficult to detect faults in such applications because often there is no ''test oracle'' to verify the correctness of the computed outputs. To help address the software quality, in this paper we present a technique for testing the implementations of machine learning classification algorithms which support such applications. Our approach is based on the technique ''metamorphic testing'', which has been shown to be effective to alleviate the oracle problem. Also presented include a case study on a real-world machine learning application framework, and a discussion of how programmers implementing machine learning algorithms can avoid the common pitfalls discovered in our study. We also conduct mutation analysis and cross-validation, which reveal that our method has high effectiveness in killing mutants, and that observing expected cross-validation result alone is not sufficiently effective to detect faults in a supervised classification program. The effectiveness of metamorphic testing is further confirmed by the detection of real faults in a popular open-source classification program. </sup></sub>---
layout: post
title: Foundations I -Andrew Ng - Nuts and Bolts of Applying Deep Learning
desc: 2017-W1
categories: 2017Course
tags:
- 0Survey
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| NIPS16 | Andrew Ng - Nuts and Bolts of Applying Deep Learning: [^1] [video](https://www.youtube.com/watch?v=F1ka6a13S9I) |
| DLSS17 | Doina Precup - Machine Learning - Bayesian Views (56:50m to 1:04:45 slides) [video](http://videolectures.net/deeplearning2017_precup_machine_learning/) + [slide](http://videolectures.net/site/normal_dl/tag=1129744/deeplearning2017_precup_machine_learning_01.pdf)|

[^1]: <sub><sup> Andrew Ng - Nuts and Bolts of Applying Deep Learning: 2016 +  a few figures about Bias/variance trade-offs </sup></sub>
![bias]({{ site.baseurl }}/pics/highBias.png){:class="img-responsive"}
![variance]({{ site.baseurl }}/pics/HighVar.png){:class="img-responsive"}
![both]({{ site.baseurl }}/pics/bothBiasVar.png){:class="img-responsive"}
![DNNflow]({{ site.baseurl }}/pics/BiasDNNflow.png){:class="img-responsive"}
---
layout: post
title: Foundations II - Ganguli - Theoretical Neuroscience and Deep Learning DLSS16
desc: 2017-W1
tags:
- 1Foundations
categories: 2017Course
---

### Ganguli - Theoretical Neuroscience and Deep Learning

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| DLSS16 | [video](http://videolectures.net/deeplearning2016_ganguli_theoretical_neuroscience/) |
| DLSS17 | [video](http://videolectures.net/deeplearning2017_ganguli_deep_learning_theory/) +  [slide](http://videolectures.net/site/normal_dl/tag=1129737/deeplearning2017_ganguli_deep_learning_theory_01.pdf)|
| DLSS17 | Deep learning in the brain | [DLSS17](http://videolectures.net/site/normal_dl/tag=1129742/deeplearning2017_richards_neuroscience_01.pdf) + [Video](http://videolectures.net/deeplearning2017_richards_neuroscience/)  |

---
layout: post
title: Reinforcement I - Pineau - RL Basic Concepts
desc: 2017-W2
tags:
- 6Reinforcement
- 0Survey
categories: 2017Course
---

### Pineau - RL Basic Concepts  

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| DLSS16 | [video](http://videolectures.net/deeplearning2016_pineau_reinforcement_learning/) |
| RLSS17 | [slideRaw](https://drive.google.com/file/d/0BzUSSMdMszk6bjl3eU5CVmU0cWs/view) + [video](http://videolectures.net/deeplearning2016_pineau_advanced_topics/)+ [slide](http://videolectures.net/site/normal_dl/tag=1137927/deeplearning2017_pineau_reinforcement_learning_01.pdf) |
---
layout: post
title: Generative I - GAN tutorial by Ian Goodfellow
desc: 2017-W2
tags:
- 5Generative
- 0Survey
categories: 2017Course
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| NIPS 2016 | ganerative adversarial network tutorial (NIPS 2016) | [paper](https://arxiv.org/abs/1701.00160) + [video](https://www.youtube.com/watch?v=AJVyzd0rqdc) + [code](https://github.com/hwalsuklee/tensorflow-generative-model-collections)|
| DLSS 2017 | Generative Models I - DLSS 2017 | [slideraw](https://drive.google.com/file/d/0ByUKRdiCDK7-bTgxTGoxYjQ4NW8/view) + [video](http://videolectures.net/deeplearning2017_goodfellow_generative_models/) + [slide](http://videolectures.net/site/normal_dl/tag=1129751/deeplearning2017_goodfellow_generative_models_01.pdf) |
---
layout: post
title: Foundations III - Investigating Behaviors of DNN
desc: 2017-W3
tags:
- 1Foundations
categories: 2017Course
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Rita | On the Expressive Power of Deep Neural Networks [^1]| [PDF](https://arxiv.org/pdf/1606.05336.pdf) | [PDF]({{site.baseurl}}/talks/20170905-Rita.pdf) |
| Arshdeep | Understanding deep learning requires rethinking generalization, ICLR17 [^2]|  [PDF](https://arxiv.org/pdf/1611.03530.pdf) | [PDF]({{site.baseurl}}/talks/20170905-Arshdeep.pdf) |
| Tianlu | On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima, ICLR17  [^3]| [PDF](https://arxiv.org/pdf/1609.04836.pdf) | [PDF]({{site.baseurl}}/talks/20170905-Tianlu.pdf) |


[^1]: <sub><sup> On the Expressive Power of Deep Neural Networks / ICML 2017/ We propose a new approach to the problem of neural network expressivity, which seeks to characterize how structural properties of a neural network family affect the functions it is able to compute. Our approach is based on an interrelated set of measures of expressivity, unified by the novel notion of trajectory length, which measures how the output of a network changes as the input sweeps along a one-dimensional path. Our findings can be summarized as follows:  (1) The complexity of the computed function grows exponentially with depth. (2) All weights are not equal: trained networks are more sensitive to their lower (initial) layer weights. (3) Regularizing on trajectory length (trajectory regularization) is a simpler alternative to batch normalization, with the same performance. </sup></sub>



[^2]: <sub><sup> Understanding deep learning requires rethinking generalization, ICLR17 / Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals (Submitted on 10 Nov 2016 (v1), last revised 26 Feb 2017 (this version, v2)) Despite their massive size, successful deep artificial neural networks can exhibit a remarkably small difference between training and test performance. Conventional wisdom attributes small generalization error either to properties of the model family, or to the regularization techniques used during training. Through extensive systematic experiments, we show how these traditional approaches fail to explain why large neural networks generalize well in practice. Specifically, our experiments establish that state-of-the-art convolutional networks for image classification trained with stochastic gradient methods easily fit a random labeling of the training data. This phenomenon is qualitatively unaffected by explicit regularization, and occurs even if we replace the true images by completely unstructured random noise. We corroborate these experimental findings with a theoretical construction showing that simple depth two neural networks already have perfect finite sample expressivity as soon as the number of parameters exceeds the number of data points as it usually does in practice. We interpret our experimental findings by comparison with traditional models. </sup></sub>


[^3]: <sub><sup> On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima, ICLR17 / Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang/ The stochastic gradient descent (SGD) method and its variants are algorithms of choice for many Deep Learning tasks. These methods operate in a small-batch regime wherein a fraction of the training data, say 32-512 data points, is sampled to compute an approximation to the gradient. It has been observed in practice that when using a larger batch there is a degradation in the quality of the model, as measured by its ability to generalize. We investigate the cause for this generalization drop in the large-batch regime and present numerical evidence that supports the view that large-batch methods tend to converge to sharp minimizers of the training and testing functions - and as is well known, sharp minima lead to poorer generalization. In contrast, small-batch methods consistently converge to flat minimizers, and our experiments support a commonly held view that this is due to the inherent noise in the gradient estimation. We discuss several strategies to attempt to help large-batch methods eliminate this generalization gap. </sup></sub>
---
layout: post
title: Foundations IV - Investigating Behaviors of DNN
desc: 2017-W3
tags:
- 1Foundations
categories: 2017Course
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Beilun | Learning Deep Parsimonious Representations, NIPS16 [^1] | [PDF](https://papers.nips.cc/paper/6263-learning-deep-parsimonious-representations) | [PDF]({{site.baseurl}}/talks/20170907-Beilun.pdf) |
| Jack | Dense Associative Memory for Pattern Recognition, NIPS16 [^2]| [PDF](https://arxiv.org/abs/1606.01164) + [video](hhttps://www.youtube.com/watch?v=30lMjQk_Lb0) | [PDF]({{site.baseurl}}/talks/20170907-Jack.pdf) |


[^1]: <sub><sup> Learning Deep Parsimonious Representations, NIPS16 / from 	Raquel Urtasun et al. / In this paper we aim at facilitating generalization for deep networks while supporting interpretability of the learned representations. Towards this goal, we propose a clustering based regularization that encourages parsimonious representations. Our k-means style objective is easy to optimize and flexible, supporting various forms of clustering, such as sample clustering, spatial clustering, as well as co-clustering. We demonstrate the effectiveness of our approach on the tasks of unsupervised learning, classification, fine grained categorization, and zero-shot learning. </sup></sub>





[^2]: <sub><sup>  Dense Associative Memory for Pattern Recognition, NIPS16 / from Hopfield/ A model of associative memory is studied, which stores and reliably retrieves many more patterns than the number of neurons in the network. We propose a simple duality between this dense associative memory and neural networks commonly used in deep learning. On the associative memory side of this duality, a family of models that smoothly interpolates between two limiting cases can be constructed. One limit is referred to as the feature-matching mode of pattern recognition, and the other one as the prototype regime. On the deep learning side of the duality, this family corresponds to feedforward neural networks with one hidden layer and various activation functions, which transmit the activities of the visible neurons to the hidden layer. This family of activation functions includes logistics, rectified linear units, and rectified polynomials of higher degrees. The proposed duality makes it possible to apply energy-based intuition from associative memory to analyze computational properties of neural networks with unusual activation functions - the higher rectified polynomials which until now have not been used in deep learning. The utility of the dense memories is illustrated for two test cases: the logical gate XOR and the recognition of handwritten digits from the MNIST data set. </sup></sub>





---
layout: post
title: Foundations V - More about Behaviors of DNN
desc: 2017-W4
tags:
- 1Foundations
categories: 2017Course
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
|  Ceyer | A Closer Look at Memorization in Deep Networks, ICML17 [^1] | [PDF](https://arxiv.org/pdf/1706.05394.pdf) | [PDF]({{site.baseurl}}/talks/20170912-Ceyer.pdf) |
|  | On the Expressive Efficiency of Overlapping Architectures of Deep Learning [^2]| [DLSSpdf](https://drive.google.com/file/d/0B6NHiPcsmak1ZzVkci1EdVN2YkU/view?usp=drive_web) + [video](http://videolectures.net/deeplearning2017_sharir_deep_learning/) |



[^1]: <sub><sup>  A Closer Look at Memorization in Deep Networks, ICML17/ from etc Aaron Courville, Yoshua Bengio, Simon Lacoste-Julien / We examine the role of memorization in deep learning, drawing connections to capacity, generalization, and adversarial robustness. While deep networks are capable of memorizing noise data, our results suggest that they tend to prioritize learning simple patterns first. In our experiments, we expose qualitative differences in gradient-based optimization of deep neural networks (DNNs) on noise vs. real data. We also demonstrate that for appropriately tuned explicit regularization (e.g., dropout) we can degrade DNN training performance on noise datasets without compromising generalization on real data. Our analysis suggests that the notions of effective capacity which are dataset independent are unlikely to explain the generalization performance of deep networks when trained with gradient based methods because training data itself plays an important role in determining the degree of memorization. </sup></sub>






[^2]: <sub><sup>  On the Expressive Efficiency of Overlapping Architectures of Deep Learning  / ICLR 2018/ Expressive efficiency refers to the relation between two architectures A and B, whereby any function realized by B could be replicated by A, but there exists functions realized by A, which cannot be replicated by B unless its size grows significantly larger. For example, it is known that deep networks are exponentially efficient with respect to shallow networks, in the sense that a shallow network must grow exponentially large in order to approximate the functions represented by a deep network of polynomial size. In this work, we extend the study of expressive efficiency to the attribute of network connectivity and in particular to the effect of "overlaps" in the convolutional process, i.e., when the stride of the convolution is smaller than its filter size (receptive field). To theoretically analyze this aspect of network's design, we focus on a well-established surrogate for ConvNets called Convolutional Arithmetic Circuits (ConvACs), and then demonstrate empirically that our results hold for standard ConvNets as well. Specifically, our analysis shows that having overlapping local receptive fields, and more broadly denser connectivity, results in an exponential increase in the expressive capacity of neural networks. Moreover, while denser connectivity can increase the expressive capacity, we show that the most common types of modern architectures already exhibit exponential increase in expressivity, without relying on fully-connected layers. </sup></sub>
---
layout: post
title: Foundations VI - More about Behaviors of DNN
desc: 2017-W4
tags:
- 1Foundations
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| SE |  Equivariance Through Parameter-Sharing, ICML17 [^1]| [PDF](https://arxiv.org/abs/1702.08389) |
| SE |  Why Deep Neural Networks for Function Approximation?, ICLR17 [^2] | [PDF](https://openreview.net/pdf?id=SkpSlKIel) |
| SE |  Geometry of Neural Network Loss Surfaces via Random Matrix Theory, [^3]ICML17 | [PDF](http://proceedings.mlr.press/v70/pennington17a/pennington17a.pdf)|
|  | Measuring Sample Quality with Kernels, NIPS16 [^4]| [PDF](https://arxiv.org/abs/1703.01717)  |
|   |  Sharp Minima Can Generalize For Deep Nets, ICML17 [^5]| [PDF](https://arxiv.org/abs/1703.04933) |


[^1]: <sub><sup>  Sharp Minima Can Generalize For Deep Nets, ICML17 / Laurent Dinh, Razvan Pascanu, Samy Bengio, Yoshua Bengio / Despite their overwhelming capacity to overfit, deep learning architectures tend to generalize relatively well to unseen data, allowing them to be deployed in practice. However, explaining why this is the case is still an open area of research. One standing hypothesis that is gaining popularity, e.g. Hochreiter & Schmidhuber (1997); Keskar et al. (2017), is that the flatness of minima of the loss function found by stochastic gradient based methods results in good generalization. This paper argues that most notions of flatness are problematic for deep models and can not be directly applied to explain generalization. Specifically, when focusing on deep networks with rectifier units, we can exploit the particular geometry of parameter space induced by the inherent symmetries that these architectures exhibit to build equivalent models corresponding to arbitrarily sharper minima. Furthermore, if we allow to reparametrize a function, the geometry of its parameters can change drastically without affecting its generalization properties. </sup></sub>


[^2]: <sub><sup>  Measuring Sample Quality with Kernels, NIPS15/ To improve the efficiency of Monte Carlo estimation, practitioners are turning to biased Markov chain Monte Carlo procedures that trade off asymptotic exactness for computational speed. The reasoning is sound: a reduction in variance due to more rapid sampling can outweigh the bias introduced. However, the inexactness creates new challenges for sampler and parameter selection, since standard measures of sample quality like effective sample size do not account for asymptotic bias. To address these challenges, we introduce a new computable quality measure based on Stein's method that quantifies the maximum discrepancy between sample and target expectations over a large class of test functions. We use our tool to compare exact, biased, and deterministic sample sequences and illustrate applications to hyperparameter selection, convergence rate assessment, and quantifying bias-variance tradeoffs in posterior inference. </sup></sub>


[^3]: <sub><sup>  Equivariance Through Parameter-Sharing, ICML17/ We propose to study equivariance in deep neural networks through parameter symmetries. In particular, given a group G that acts discretely on the input and output of a standard neural network layer ϕW:ℜM→ℜN, we show that ϕW is equivariant with respect to G-action iff G explains the symmetries of the network parameters W. Inspired by this observation, we then propose two parameter-sharing schemes to induce the desirable symmetry on W. Our procedures for tying the parameters achieve G-equivariance and, under some conditions on the action of , they guarantee sensitivity to all other permutation groups outside. </sup></sub>


[^4]: <sub><sup>  Why Deep Neural Networks for Function Approximation?, ICLR17 / Recently there has been much interest in understanding why deep neural networks are preferred to shallow networks. We show that, for a large class of piecewise smooth functions, the number of neurons needed by a shallow network to approximate a function is exponentially larger than the corresponding number of neurons needed by a deep network for a given degree of function approximation. First, we consider univariate functions on a bounded interval and require a neural network to achieve an approximation error of ε uniformly over the interval. We show that shallow networks (i.e., networks whose depth does not depend on ε) require Ω(poly(1/ε)) neurons while deep networks (i.e., networks whose depth grows with 1/ε) require O(polylog(1/ε)) neurons. We then extend these results to certain classes of important multivariate functions. Our results are derived for neural networks which use a combination of rectifier linear units (ReLUs) and binary step units, two of the most popular type of activation functions. Our analysis builds on a simple observation: the multiplication of two bits can be represented by a ReLU. </sup></sub>


[^5]: <sub><sup>  Geometry of Neural Network Loss Surfaces via Random Matrix Theory, ICML17 / Understanding the geometry of neural network loss surfaces is important for the development of improved optimization algorithms and for building a theoretical understanding of why deep learning works. In this paper, we study the geometry in terms of the distribution of eigenvalues of the Hessian matrix at critical points of varying energy. We introduce an analytical framework and a set of tools from random matrix theory that allow us to compute an approximation of this distribution under a set of simplifying assumptions. The shape of the spectrum depends strongly on the energy and another key parameter, ϕ, which measures the ratio of parameters to data points. Our analysis predicts and numerical simulations support that for critical points of small index, the number of negative eigenvalues scales like the 3/2 power of the energy. We leave as an open problem an explanation for our observation that, in the context of a certain memorization task, the energy of minimizers is well-approximated by the function 1/2(1−ϕ)^2. </sup></sub>---
layout: post
title: Structure I - Varying DNN structures for an application
desc: 2017-W5
tags:
- 2Structures
categories: 2017Course
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Jack | Learning End-to-End Goal-Oriented Dialog, ICLR17 [^1] | [PDF](https://arxiv.org/abs/1605.07683) | [PDF]({{site.baseurl}}/talks/20170919-Jack.pdf) |
| Bargav | Nonparametric Neural Networks, ICLR17 [^2]| [PDF](https://openreview.net/pdf?id=BJK3Xasel) | [PDF]({{site.baseurl}}/talks/20170919-Bargav.pdf) |
| Bargav | Learning Structured Sparsity in Deep Neural Networks, NIPS16 [^3]| [PDF](https://arxiv.org/abs/1608.03665) | [PDF]({{site.baseurl}}/talks/20170912-Bargav.pdf) |
| Arshdeep | Learning the Number of Neurons in Deep Networks, NIPS16 [^4]| [PDF](https://papers.nips.cc/paper/6372-learning-the-number-of-neurons-in-deep-networks) | [PDF]({{site.baseurl}}/talks/20170912-Arshdeep.pdf) |





[^1]: <sub><sup> Learning End-to-End Goal-Oriented Dialog, ICLR17 / Antoine Bordes, Y-Lan Boureau, Jason Weston/ Traditional dialog systems used in goal-oriented applications require a lot of domain-specific handcrafting, which hinders scaling up to new domains. End-to-end dialog systems, in which all components are trained from the dialogs themselves, escape this limitation. But the encouraging success recently obtained in chit-chat dialog may not carry over to goal-oriented settings. This paper proposes a testbed to break down the strengths and shortcomings of end-to-end dialog systems in goal-oriented applications. Set in the context of restaurant reservation, our tasks require manipulating sentences and symbols, so as to properly conduct conversations, issue API calls and use the outputs of such calls. We show that an end-to-end dialog system based on Memory Networks can reach promising, yet imperfect, performance and learn to perform non-trivial operations. We confirm those results by comparing our system to a hand-crafted slot-filling baseline on data from the second Dialog State Tracking Challenge (Henderson et al., 2014a). We show similar result patterns on data extracted from an online concierge service. </sup></sub>


[^2]: <sub><sup>  Nonparametric Neural Networks, ICLR17 / George Philipp, Jaime G. Carbonell/ Automatically determining the optimal size of a neural network for a given task without prior information currently requires an expensive global search and training many networks from scratch. In this paper, we address the problem of automatically finding a good network size during a single training cycle. We introduce *nonparametric neural networks*, a non-probabilistic framework for conducting optimization over all possible network sizes and prove its soundness when network growth is limited via an L_p penalty. We train networks under this framework by continuously adding new units while eliminating redundant units via an L_2 penalty. We employ a novel optimization algorithm, which we term *adaptive radial-angular gradient descent* or *AdaRad*, and obtain promising results. </sup></sub>


[^3]: <sub><sup> Learning Structured Sparsity in Deep Neural Networks, NIPS16/ High demand for computation resources severely hinders deployment of large-scale Deep Neural Networks (DNN) in resource constrained devices. In this work, we propose a Structured Sparsity Learning (SSL) method to regularize the structures (i.e., filters, channels, filter shapes, and layer depth) of DNNs. SSL can: (1) learn a compact structure from a bigger DNN to reduce computation cost; (2) obtain a hardware-friendly structured sparsity of DNN to efficiently accelerate the DNN's evaluation. Experimental results show that SSL achieves on average 5.1 × and 3.1 × speedups of convolutional layer computation of AlexNet against CPU and GPU, respectively, with off-the-shelf libraries. These speedups are about twice speedups of non-structured sparsity; (3) regularize the DNN structure to improve classification accuracy. The results show that for CIFAR-10, regularization on layer depth reduces a 20-layer Deep Residual Network (ResNet) to 18 layers while improves the accuracy from 91.25% to 92.60%, which is still higher than that of original ResNet with 32 layers. For AlexNet, SSL reduces the error by ~ 1%. </sup></sub>


[^4]: <sub><sup> Learning the Number of Neurons in Deep Networks, NIPS16 / Nowadays, the number of layers and of neurons in each layer of a deep network are typically set manually. While very deep and wide networks have proven effective in general, they come at a high memory and computation cost, thus making them impractical for constrained platforms. These networks, however, are known to have many redundant parameters, and could thus, in principle, be replaced by more compact architectures. In this paper, we introduce an approach to automatically determining the number of neurons in each layer of a deep network during learning. To this end, we propose to make use of a group sparsity regularizer on the parameters of the network, where each group is defined to act on a single neuron. Starting from an overcomplete network, we show that our approach can reduce the number of parameters by up to 80\% while retaining or even improving the network accuracy. </sup></sub>

---
layout: post
title: Structure II - DNN with Varying Structures
desc: 2017-W5
tags:
- 2Structures
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Shijia | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, (Dean), ICLR17 [^1]| [PDF](https://arxiv.org/abs/1701.06538) | [PDF]({{site.baseurl}}/talks/20170921-Shijia.pdf) |
| Ceyer | Sequence Modeling via Segmentations, ICML17 [^2]| [PDF](https://arxiv.org/abs/1702.07463) | [PDF]({{site.baseurl}}/talks/20170921-Ceyer.pdf) |
| Arshdeep | Input Switched Affine Networks: An RNN Architecture Designed for Interpretability, ICML17 [^3]|  [PDF](http://proceedings.mlr.press/v70/foerster17a/foerster17a.pdf) | [PDF]({{site.baseurl}}/talks/20170921-Arshdeep.pdf) |



[^1]: <sub><sup>  Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, (Dean), ICLR17/ The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation. In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.
</sup></sub>



[^2]: <sub><sup>  Sequence Modeling via Segmentations, ICML17/ Segmental structure is a common pattern in many types of sequences such as phrases in human languages. In this paper, we present a probabilistic model for sequences via their segmentations. The probability of a segmented sequence is calculated as the product of the probabilities of all its segments, where each segment is modeled using existing tools such as recurrent neural networks. Since the segmentation of a sequence is usually unknown in advance, we sum over all valid segmentations to obtain the final probability for the sequence. An efficient dynamic programming algorithm is developed for forward and backward computations without resorting to any approximation. We demonstrate our approach on text segmentation and speech recognition tasks. In addition to quantitative results, we also show that our approach can discover meaningful segments in their respective application contexts. </sup></sub>




[^3]: <sub><sup> Input Switched Affine Networks: An RNN Architecture Designed for Interpretability, ICML17/ There exist many problem domains where the interpretability of neural network models is essential for deployment. Here we introduce a recurrent architecture composed of input-switched affine transformations - in other words an RNN without any explicit nonlinearities, but with input-dependent recurrent weights. This simple form allows the RNN to be analyzed via straightforward linear methods: we can exactly characterize the linear contribution of each input to the model predictions; we can use a change-of-basis to disentangle input, output, and computational hidden unit subspaces; we can fully reverse-engineer the architecture's solution to a simple task. Despite this ease of interpretation, the input switched affine network achieves reasonable performance on a text modeling tasks, and allows greater computational efficiency than networks with standard nonlinearities. </sup></sub>---
layout: post
title: Structure III - DNN with Attention
desc: 2017-W6
tags:
- 2Structures
categories: 2017Course
---




| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Rita | Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, ICLR17 [^1] |  [PDF](https://arxiv.org/abs/1612.03928)| [PDF]({{site.baseurl}}/talks/20170926-Rita.pdf) |
| Tianlu  | Dynamic Coattention Networks For Question Answering, ICLR17 [^2]| [PDF](https://arxiv.org/abs/1611.01604) + [code](https://github.com/marshmelloX/dynamic-coattention-network)| [PDF]({{site.baseurl}}/talks/20170926-Tianlu.pdf) |
| ChaoJiang | Structured Attention Networks, ICLR17 [^3] |[PDF](https://arxiv.org/abs/1702.00887) + [code](https://github.com/harvardnlp/struct-attn) | [PDF]({{site.baseurl}}/talks/20170928-Chao.pdf) |


[^1]: <sub><sup>  Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, ICLR17 / Attention plays a critical role in human visual experience. Furthermore, it has recently been demonstrated that attention can also play an important role in the context of applying artificial neural networks to a variety of tasks from fields such as computer vision and NLP. In this work we show that, by properly defining attention for convolutional neural networks, we can actually use this type of information in order to significantly improve the performance of a student CNN network by forcing it to mimic the attention maps of a powerful teacher network. To that end, we propose several novel methods of transferring attention, showing consistent improvement across a variety of datasets and convolutional neural network architectures. Code and models for our experiments are available at this https [URL](https://github.com/szagoruyko/attention-transfer). </sup></sub>



[^2]: <sub><sup>  Dynamic Coattention Networks For Question Answering, ICLR17 / Caiming Xiong, Victor Zhong, Richard Socher/ Several deep learning models have been proposed for question answering. However, due to their single-pass nature, they have no way to recover from local maxima corresponding to incorrect answers. To address this problem, we introduce the Dynamic Coattention Network (DCN) for question answering. The DCN first fuses co-dependent representations of the question and the document in order to focus on relevant parts of both. Then a dynamic pointing decoder iterates over potential answer spans. This iterative procedure enables the model to recover from initial local maxima corresponding to incorrect answers. On the Stanford question answering dataset, a single DCN model improves the previous state of the art from 71.0% F1 to 75.9%, while a DCN ensemble obtains 80.4% F1. </sup></sub>


[^3]: <sub><sup>  Structured Attention Networks, ICLR17 / Attention networks have proven to be an effective approach for embedding categorical inference within a deep neural network. However, for many tasks we may want to model richer structural dependencies without abandoning end-to-end training. In this work, we experiment with incorporating richer structural distributions, encoded using graphical models, within deep networks. We show that these structured attention networks are simple extensions of the basic attention procedure, and that they allow for extending attention beyond the standard soft-selection approach, such as attending to partial segmentations or to subtrees. We experiment with two different classes of structured attention networks: a linear-chain conditional random field and a graph-based parsing model, and describe how these models can be practically implemented as neural network layers. Experiments show that this approach is effective for incorporating structural biases, and structured attention networks outperform baseline attention models on a variety of synthetic and real tasks: tree transduction, neural machine translation, question answering, and natural language inference. We further find that models trained in this way learn interesting unsupervised hidden representations that generalize simple attention. </sup></sub>---
layout: post
title: Structure IV - DNN with Attention 2
desc: 2017-W6
tags:
- 2Structures
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Jack  |  Attend, Adapt and Transfer: Attentive Deep Architecture for Adaptive Transfer from multiple sources in the same domain, ICLR17 [^1]| [PDF](https://arxiv.org/abs/1510.02879)| [PDF]({{site.baseurl}}/talks/20170928-Jack.pdf) |
| Arshdeep | Bidirectional Attention Flow for Machine Comprehension, ICLR17 [^2] | [PDF](https://arxiv.org/abs/1611.01603) + [code](https://github.com/allenai/bi-att-flow)| [PDF]({{site.baseurl}}/talks/20170928-Arshdeep.pdf) |
| Ceyer | Image-to-Markup Generation with Coarse-to-Fine Attention, ICML17 |[PDF](http://lstm.seas.harvard.edu/latex/) + [code](https://github.com/harvardnlp/im2markup) | [PDF]({{site.baseurl}}/talks/20170928-Ceyer.pdf) |
| ChaoJiang |  Can Active Memory Replace Attention? ; Samy Bengio, NIPS16 [^3]| [PDF](https://arxiv.org/abs/1610.08613)  | [PDF]({{site.baseurl}}/talks/20171003-Chao.pdf) |
|  | An Information-Theoretic Framework for Fast and Robust Unsupervised Learning via Neural Population Infomax, ICLR17 | [PDF](https://arxiv.org/abs/1611.01886)|



[^1]: <sub><sup>  Bidirectional Attention Flow for Machine Comprehension, ICLR17 /Machine comprehension (MC), answering a query about a given context paragraph, requires modeling complex interactions between the context and the query. Recently, attention mechanisms have been successfully extended to MC. Typically these methods use attention to focus on a small portion of the context and summarize it with a fixed-size vector, couple attentions temporally, and/or often form a uni-directional attention. In this paper we introduce the Bi-Directional Attention Flow (BIDAF) network, a multi-stage hierarchical process that represents the context at different levels of granularity and uses bi-directional attention flow mechanism to obtain a query-aware context representation without early summarization. Our experimental evaluations show that our model achieves the state-of-the-art results in Stanford Question Answering Dataset (SQuAD) and CNN/DailyMail cloze test. </sup></sub>



[^2]: <sub><sup>  Image-to-Markup Generation with Coarse-to-Fine Attention, ICML17/ We present a neural encoder-decoder model to convert images into presentational markup based on a scalable coarse-to-fine attention mechanism. Our method is evaluated in the context of image-to-LaTeX generation, and we introduce a new dataset of real-world rendered mathematical expressions paired with LaTeX markup. We show that unlike neural OCR techniques using CTC-based models, attention-based approaches can tackle this non-standard OCR task. Our approach outperforms classical mathematical OCR systems by a large margin on in-domain rendered data, and, with pretraining, also performs well on out-of-domain handwritten data. To reduce the inference complexity associated with the attention-based approaches, we introduce a new coarse-to-fine attention layer that selects a support region before applying attention. </sup></sub>


[^3]: <sub><sup>  Can Active Memory Replace Attention? ; Samy Bengio, NIPS16/ Several mechanisms to focus attention of a neural network on selected parts of its input or memory have been used successfully in deep learning models in recent years. Attention has improved image classification, image captioning, speech recognition, generative models, and learning algorithmic tasks, but it had probably the largest impact on neural machine translation. Recently, similar improvements have been obtained using alternative mechanisms that do not focus on a single part of a memory but operate on all of it in parallel, in a uniform way. Such mechanism, which we call active memory, improved over attention in algorithmic tasks, image processing, and in generative modelling. So far, however, active memory has not improved over attention for most natural language processing tasks, in particular for machine translation. We analyze this shortcoming in this paper and propose an extended model of active memory that matches existing attention models on neural machine translation and generalizes better to longer sentences. We investigate this model and explain why previous active memory models did not succeed. Finally, we discuss when active memory brings most benefits and where attention can be a better choice. </sup></sub>
---
layout: post
title: Structure V - DNN with Attention 3
desc: 2017-W7
tags:
- 2Structures
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Tianlu | Ask Me Anything: Dynamic Memory Networks for Natural Language Processing, ICML17 [^1]| [PDF](https://arxiv.org/abs/1506.07285) + [code](https://github.com/swstarlab/DynamicMemoryNetworks) | [PDF]({{site.baseurl}}/talks/20171003-Tianlu.pdf) |
| Jack | Reasoning with Memory Augmented Neural Networks for Language Comprehension, ICLR17 [^2]| [PDF](https://arxiv.org/abs/1610.06454) | [PDF]({{site.baseurl}}/talks/20171003-jack.pdf) |
| Xueying | State-Frequency Memory Recurrent Neural Networks, ICML17 [^3] | [PDF](http://proceedings.mlr.press/v70/hu17c.html) | [PDF]({{site.baseurl}}/talks/20171003-Xueying.pdf) |


[^1]: <sub><sup>  Ask Me Anything: Dynamic Memory Networks for Natural Language Processing, ICML16 / Most tasks in natural language processing can be cast into question answering (QA) problems over language input. We introduce the dynamic memory network (DMN), a neural network architecture which processes input sequences and questions, forms episodic memories, and generates relevant answers. Questions trigger an iterative attention process which allows the model to condition its attention on the inputs and the result of previous iterations. These results are then reasoned over in a hierarchical recurrent sequence model to generate answers. The DMN can be trained end-to-end and obtains state-of-the-art results on several types of tasks and datasets: question answering (Facebook's bAbI dataset), text classification for sentiment analysis (Stanford Sentiment Treebank) and sequence modeling for part-of-speech tagging (WSJ-PTB). The training for these different tasks relies exclusively on trained word vector representations and input-question-answer triplets. (high citation) </sup></sub>



[^2]: <sub><sup>  Reasoning with Memory Augmented Neural Networks for Language Comprehension, ICLR17 / Hypothesis testing is an important cognitive process that supports human reasoning. In this paper, we introduce a computational hypothesis testing approach based on memory augmented neural networks. Our approach involves a hypothesis testing loop that reconsiders and progressively refines a previously formed hypothesis in order to generate new hypotheses to test. We apply the proposed approach to language comprehension task by using Neural Semantic Encoders (NSE). Our NSE models achieve the state-of-the-art results showing an absolute improvement of 1.2% to 2.6% accuracy over previous results obtained by single and ensemble systems on standard machine comprehension benchmarks such as the Children's Book Test (CBT) and Who-Did-What (WDW) news article datasets. </sup></sub>


[^3]: <sub><sup>  State-Frequency Memory Recurrent Neural Networks, ICML17/ Modeling temporal sequences plays a fundamental role in various modern applications and has drawn more and more attentions in the machine learning community. Among those efforts on improving the capability to represent temporal data, the Long Short-Term Memory (LSTM) has achieved great success in many areas. Although the LSTM can capture long-range dependency in the time domain, it does not explicitly model the pattern occurrences in the frequency domain that plays an important role in tracking and predicting data points over various time cycles. We propose the State-Frequency Memory (SFM), a novel recurrent architecture that allows to separate dynamic patterns across different frequency components and their impacts on modeling the temporal contexts of input sequences. By jointly decomposing memorized dynamics into state-frequency components, the SFM is able to offer a fine-grained analysis of temporal sequences by capturing the dependency of uncovered patterns in both time and frequency domains. Evaluations on several temporal modeling tasks demonstrate the SFM can yield competitive performances, in particular as compared with the state-of-the-art LSTM models. </sup></sub>---
layout: post
title: Structure VI - DNN with Adaptive Structures
desc: 2017-W7
tags:
- 2Structures
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Anant | AdaNet: Adaptive Structural Learning of Artificial Neural Networks, ICML17 [^1] | [PDF](https://arxiv.org/abs/1607.01097) | [PDF]({{site.baseurl}}/talks/20171005-Anant.pdf) |
|Shijia | SplitNet: Learning to Semantically Split Deep Networks for Parameter Reduction and Model Parallelization, ICML17 [^2]| [PDF](http://proceedings.mlr.press/v70/kim17b/kim17b.pdf) | [PDF]({{site.baseurl}}/talks/20171005-Shijia.pdf) |
| Jack | Proximal Deep Structured Models, NIPS16  [^3]| [PDF](https://papers.nips.cc/paper/6074-proximal-deep-structured-models) | [PDF]({{site.baseurl}}/talks/20171010-Jack.pdf) |
|  | Optimal Architectures in a Solvable Model of Deep Networks, NIPS16 [^4] | [PDF](https://papers.nips.cc/paper/6330-optimal-architectures-in-a-solvable-model-of-deep-networks) |
| Tianlu | Large-Scale Evolution of Image Classifiers, ICML17 [^5]|[PDF](https://arxiv.org/abs/1703.01041)  | [PDF]({{site.baseurl}}/talks/20170912-Tianlu.pdf) |


[^1]: <sub><sup>  Large-Scale Evolution of Image Classifiers, ICML17/ Esteban Real, Sherry Moore, Andrew Selle, Saurabh Saxena, Yutaka Leon Suematsu, Jie Tan, Quoc Le, Alex Kurakin / Neural networks have proven effective at solving difficult problems but designing their architectures can be challenging, even for image classification problems alone. Our goal is to minimize human participation, so we employ evolutionary algorithms to discover such networks automatically. Despite significant computational requirements, we show that it is now possible to evolve models with accuracies within the range of those published in the last year. Specifically, we employ simple evolutionary techniques at unprecedented scales to discover models for the CIFAR-10 and CIFAR-100 datasets, starting from trivial initial conditions and reaching accuracies of 94.6% (95.6% for ensemble) and 77.0%, respectively. To do this, we use novel and intuitive mutation operators that navigate large search spaces; we stress that no human participation is required once evolution starts and that the output is a fully-trained model. Throughout this work, we place special emphasis on the repeatability of results, the variability in the outcomes and the computational requirements. </sup></sub>



[^2]: <sub><sup> AdaNet: Adaptive Structural Learning of Artificial Neural Networks, ICML17 / Corinna Cortes, et al.  We present new algorithms for adaptively learning artificial neural networks. Our algorithms (AdaNet) adaptively learn both the structure of the network and its weights. They are based on a solid theoretical analysis, including data-dependent generalization guarantees that we prove and discuss in detail. We report the results of large-scale experiments with one of our algorithms on several binary classification tasks extracted from the CIFAR-10 dataset. The results demonstrate that our algorithm can automatically learn network structures with very competitive performance accuracies when compared with those achieved for neural networks found by standard approaches. </sup></sub>


[^3]: <sub><sup>  SplitNet: Learning to Semantically Split Deep Networks for Parameter Reduction and Model Parallelization, ICML17 / We propose a novel deep neural network that is both lightweight and effectively structured for model parallelization. Our network, which we name as SplitNet, automatically learns to split the network weights into either a set or a hierarchy of multiple groups that use disjoint sets of features, by learning both the class-to-group and feature-to-group assignment matrices along with the network weights. This produces a tree-structured network that involves no connection between branched subtrees of semantically disparate class groups. SplitNet thus greatly reduces the number of parameters and requires significantly less computations, and is also embarrassingly model parallelizable at test time, since the network evaluation for each subnetwork is completely independent except for the shared lower layer weights that can be duplicated over multiple processors. We validate our method with two deep network models (ResNet and AlexNet) on two different datasets (CIFAR-100 and ILSVRC 2012) for image classification, on which our method obtains networks with significantly reduced number of parameters while achieving comparable or superior classification accuracies over original full deep networks, and accelerated test speed with multiple GPUs. </sup></sub>


[^4]: <sub><sup>  Proximal Deep Structured Models, NIPS16 / 	Raquel Urtasun et al.  Many problems in real-world applications involve predicting continuous-valued random variables that are statistically related. In this paper, we propose a powerful deep structured model that is able to learn complex non-linear functions which encode the dependencies between continuous output variables. We show that inference in our model using proximal methods can be efficiently solved as a feedfoward pass of a special type of deep recurrent neural network. We demonstrate the effectiveness of our approach in the tasks of image denoising, depth refinement and optical flow estimation. </sup></sub>



[^5]: <sub><sup> Optimal architectures in a solvable model of Deep networks, NIPS16/ Deep neural networks have received a considerable attention due to the success of their training for real world machine learning applications. They are also of great interest to the understanding of sensory processing in cortical sensory hierarchies. The purpose of this work is to advance our theoretical understanding of the computational benefits of these architectures. Using a simple model of clustered noisy inputs and a simple learning rule, we provide analytically derived recursion relations describing the propagation of the signals along the deep network. By analysis of these equations, and defining performance measures, we show that these model networks have optimal depths. We further explore the dependence of the optimal architecture on the system parameters. </sup></sub>---
layout: post
title: Reliable Applications I - Understanding
desc: 2017-W8
tags:
- 3Reliable
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Rita | Learning Important Features Through Propagating Activation Differences, ICML17 [^1]| [PDF](https://arxiv.org/abs/1704.02685) | [PDF]({{site.baseurl}}/talks/20171010-Rita.pdf) |
| GaoJi  | Examples are not Enough, Learn to Criticize! Model Criticism for Interpretable Machine Learning, NIPS16 [^2]| [PDF](http://people.csail.mit.edu/beenkim/papers/KIM2016NIPS_MMD.pdf) | [PDF]({{site.baseurl}}/talks/20171010-Ji.pdf) |
| Rita | Learning Kernels with Random Features, Aman Sinha*; John Duchi, [^3] | [PDF](https://stanford.edu/~jduchi/projects/SinhaDu16.pdf) | [PDF]({{site.baseurl}}/talks/20170907-Rita.pdf) |


[^1]: <sub><sup> Learning Kernels with Random Features / John Duchi NIPS2016/ Randomized features provide a computationally efficient way to approximate kernel machines in machine learning tasks. However, such methods require a user-defined kernel as input. We extend the randomized-feature approach to the task of learning a kernel (via its associated random features). Specifically, we present an efficient optimization problem that learns a kernel in a supervised manner. We prove the consistency of the estimated kernel as well as generalization bounds for the class of estimators induced by the optimized kernel, and we experimentally evaluate our technique on several datasets. Our approach is efficient and highly scalable, and we attain competitive results with a fraction of the training cost of other techniques. </sup></sub>



[^2]: <sub><sup>  Learning Important Features Through Propagating Activation Differences, ICML17 / DeepLIFE / The purported "black box"' nature of neural networks is a barrier to adoption in applications where interpretability is essential. Here we present DeepLIFT (Deep Learning Important FeaTures), a method for decomposing the output prediction of a neural network on a specific input by backpropagating the contributions of all neurons in the network to every feature of the input. DeepLIFT compares the activation of each neuron to its 'reference activation' and assigns contribution scores according to the difference. By optionally giving separate consideration to positive and negative contributions, DeepLIFT can also reveal dependencies which are missed by other approaches. Scores can be computed efficiently in a single backward pass. We apply DeepLIFT to models trained on MNIST and simulated genomic data, and show significant advantages over gradient-based methods. A detailed video tutorial on the method is at this http URL and code is at this http URL. </sup></sub>


[^3]: <sub><sup>  Examples are not Enough, Learn to Criticize! Model Criticism for Interpretable Machine Learning, NIPS16 / Been Kim et al. / Example-based explanations are widely used in the effort to improve the interpretability of highly complex distributions. However, prototypes alone are rarely sufficient to represent the gist of the complexity. In order for users to construct better mental models and understand complex data distributions, we also need criticism to explain what are not captured by prototypes. Motivated by the Bayesian model criticism framework, we develop MMD-critic which efficiently learns prototypes and criticism, designed to aid human interpretability. A human subject pilot study shows that the MMD-critic selects prototypes and criticism that are useful to facilitate human understanding and reasoning. We also evaluate the prototypes selected by MMD-critic via a nearest prototype classifier, showing competitive performance compared to baselines. </sup></sub>
---
layout: post
title: Reliable Applications V - Understanding2
desc: 2017-W10
tags:
- 3Reliable
categories: 2017Course
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| ChaoJiang |  Toward Deeper Understanding of Neural Networks: The Power of Initialization and a Dual View on Expressivity, NIPS16 [^1]| [PDF](https://arxiv.org/abs/1602.05897)| [PDF]({{site.baseurl}}/talks/20171024-Chao.pdf) |
| Rita | Visualizing Deep Neural Network Decisions: Prediction Difference Analysis, ICLR17 [^2] | [PDF](https://arxiv.org/abs/1702.04595) | [PDF]({{site.baseurl}}/talks/20171024-Rita.pdf) |
| Arshdeep | Axiomatic Attribution for Deep Networks, ICML17 [^3] | [PDF](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) | [PDF]({{site.baseurl}}/talks/20171031-Arshdeep.pdf) |





[^1]: <sub><sup> Visualizing Deep Neural Network Decisions: Prediction Difference Analysis, ICLR17 / Luisa M Zintgraf, Taco S Cohen, Tameem Adel, Max Welling/ This article presents the prediction difference analysis method for visualizing the response of a deep neural network to a specific input. When classifying images, the method highlights areas in a given input image that provide evidence for or against a certain class. It overcomes several shortcoming of previous methods and provides great additional insight into the decision making process of classifiers. Making neural network decisions interpretable through visualization is important both to improve models and to accelerate the adoption of black-box classifiers in application areas such as medicine. We illustrate the method in experiments on natural images (ImageNet data), as well as medical images (MRI brain scans). </sup></sub>



[^2]: <sub><sup> Axiomatic Attribution for Deep Networks, ICML17 / Google / We study the problem of attributing the prediction of a deep network to its input features, a problem previously studied by several other works. We identify two fundamental axioms---Sensitivity and Implementation Invariance that attribution methods ought to satisfy. We show that they are not satisfied by most known attribution methods, which we consider to be a fundamental weakness of those methods. We use the axioms to guide the design of a new attribution method called Integrated Gradients. Our method requires no modification to the original network and is extremely simple to implement; it just needs a few calls to the standard gradient operator. We apply this method to a couple of image models, a couple of text models and a chemistry model, demonstrating its ability to debug networks, to extract rules from a network, and to enable users to engage with models better. </sup></sub>


[^3]: <sub><sup>  Toward Deeper Understanding of Neural Networks: The Power of Initialization and a Dual View on Expressivity, NIPS16 / GoogleBrain / We develop a general duality between neural networks and compositional kernel Hilbert spaces. We introduce the notion of a computation skeleton, an acyclic graph that succinctly describes both a family of neural networks and a kernel space. Random neural networks are generated from a skeleton through node replication followed by sampling from a normal distribution to assign weights. The kernel space consists of functions that arise by compositions, averaging, and non-linear transformations governed by the skeleton's graph topology and activation functions. We prove that random networks induce representations which approximate the kernel space. In particular, it follows that random weight initialization often yields a favorable starting point for optimization despite the worst-case intractability of training neural networks. </sup></sub>
---
layout: post
title: Reliable Applications II - Data privacy  
desc: 2017-W8
tags:
- 3Reliable
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Xueying | Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data, ICLR17 [^1]| [PDF](https://arxiv.org/abs/1610.05755) | [PDF]({{site.baseurl}}/talks/20171012-Xueying.pdf) |
| Bargav |  Deep Learning with Differential Privacy, CCS16 [^2]| [PDF](https://arxiv.org/abs/1607.00133) + [video](https://www.youtube.com/watch?v=ZxDBEyjiPxI) | [PDF]({{site.baseurl}}/talks/20171012-Bargav-2.pdf) |
| Bargav | Privacy-Preserving Deep Learning, CCS15 [^3]| [PDF](https://www.cs.cornell.edu/~shmat/shmat_ccs15.pdf) | [PDF]({{site.baseurl}}/talks/20171012-Bargav-1.pdf) |
| Xueying | Domain Separation Networks, NIPS16 [^4]| [PDF](https://arxiv.org/abs/1608.06019) | [PDF]({{site.baseurl}}/talks/20171024-Xueying.pdf) |




[^1]: <sub><sup> Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data, ICLR17 / Nicolas Papernot et al.  Some machine learning applications involve training data that is sensitive, such as the medical histories of patients in a clinical trial. A model may inadvertently and implicitly store some of its training data; careful analysis of the model may therefore reveal sensitive information. To address this problem, we demonstrate a generally applicable approach to providing strong privacy guarantees for training data: Private Aggregation of Teacher Ensembles (PATE). The approach combines, in a black-box fashion, multiple models trained with disjoint datasets, such as records from different subsets of users. Because they rely directly on sensitive data, these models are not published, but instead used as "teachers" for a "student" model. The student learns to predict an output chosen by noisy voting among all of the teachers, and cannot directly access an individual teacher or the underlying data or parameters. The student's privacy properties can be understood both intuitively (since no single teacher and thus no single dataset dictates the student's training) and formally, in terms of differential privacy. These properties hold even if an adversary can not only query the student but also inspect its internal workings. Compared with previous work, the approach imposes only weak assumptions on how teachers are trained: it applies to any model, including non-convex models like DNNs. We achieve state-of-the-art privacy/utility trade-offs on MNIST and SVHN thanks to an improved privacy analysis and semi-supervised learning. </sup></sub>



[^2]: <sub><sup>  Domain Separation Networks, NIPS16 / The cost of large scale data collection and annotation often makes the application of machine learning algorithms to new tasks or datasets prohibitively expensive. One approach circumventing this cost is training models on synthetic data where annotations are provided automatically. Despite their appeal, such models often fail to generalize from synthetic to real images, necessitating domain adaptation algorithms to manipulate these models before they can be successfully applied. Existing approaches focus either on mapping representations from one domain to the other, or on learning to extract features that are invariant to the domain from which they were extracted. However, by focusing only on creating a mapping or shared representation between the two domains, they ignore the individual characteristics of each domain. We hypothesize that explicitly modeling what is unique to each domain can improve a model's ability to extract domain-invariant features. Inspired by work on private-shared component analysis, we explicitly learn to extract image representations that are partitioned into two subspaces: one component which is private to each domain and one which is shared across domains. Our model is trained to not only perform the task we care about in the source domain, but also to use the partitioned representation to reconstruct the images from both domains. Our novel architecture results in a model that outperforms the state-of-the-art on a range of unsupervised domain adaptation scenarios and additionally produces visualizations of the private and shared representations enabling interpretation of the domain adaptation process. </sup></sub>


[^3]: <sub><sup> Deep Learning with Differential Privacy, CCS 2016/ Machine learning techniques based on neural networks are achieving remarkable results in a wide variety of domains. Often, the training of models requires large, representative datasets, which may be crowdsourced and contain sensitive information. The models should not expose private information in these datasets. Addressing this goal, we develop new algorithmic techniques for learning and a refined analysis of privacy costs within the framework of differential privacy. Our implementation and experiments demonstrate that we can train deep neural networks with non-convex objectives, under a modest privacy budget, and at a manageable cost in software complexity, training efficiency, and model quality. </sup></sub>



[^4]: <sub><sup> Privacy-Preserving Deep Learning, CCS15/ Deep learning based on artificial neural networks is a very popular approach to modeling, classifying, and recognizing complex data such as images, speech, and text. The unprecedented accuracy of deep learning methods has turned them into the foundation of new AI-based services on the Internet. Commercial companies that collect user data on a large scale have been the main beneficiaries of this trend since the success of deep learning techniques is directly proportional to the amount of data available for training. Massive data collection required for deep learning presents obvious privacy issues. Users' personal, highly sensitive data such as photos and voice recordings is kept indefinitely by the companies that collect it. Users can neither delete it, nor restrict the purposes for which it is used. Furthermore, centrally kept data is subject to legal subpoenas and extra-judicial surveillance. Many data owners--for example, medical institutions that may want to apply deep learning methods to clinical records--are prevented by privacy and confidentiality concerns from sharing the data and thus benefitting from large-scale deep learning. In this paper, we design, implement, and evaluate a practical system that enables multiple parties to jointly learn an accurate neural-network model for a given objective without sharing their input datasets. We exploit the fact that the optimization algorithms used in modern deep learning, namely, those based on stochastic gradient descent, can be parallelized and executed asynchronously. Our system lets participants train independently on their own datasets and selectively share small subsets of their models' key parameters during training. This offers an attractive point in the utility/privacy tradeoff space: participants preserve the privacy of their respective data while still benefitting from other participants' models and thus boosting their learning accuracy beyond what is achievable solely on their own inputs. We demonstrate the accuracy of our privacy-preserving deep learning on benchmark datasets. </sup></sub>---
layout: post
title: Reliable Applications III - Interesting Tasks
desc: 2017-W9
tags:
- 3Reliable
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Jack | Learning to Query, Reason, and Answer Questions On Ambiguous Texts, ICLR17 [^1]| [PDF](https://web.eecs.umich.edu/~baveja/Papers/GuoICLR2017.pdf) | [PDF]({{site.baseurl}}/talks/20171017-Jack.pdf) |
| Arshdeep |  Making Neural Programming Architectures Generalize via Recursion, ICLR17 [^2]| [PDF](https://arxiv.org/abs/1704.06611) | [PDF]({{site.baseurl}}/talks/20170919-Arshdeep.pdf) |
| Xueying | Towards Deep Interpretability (MUS-ROVER II): Learning Hierarchical Representations of Tonal Music, ICLR17 [^3] | [PDF](https://openreview.net/pdf?id=ryhqQFKgl) | [PDF]({{site.baseurl}}/talks/20170921-Xueying.pdf) |



[^1]: <sub><sup> Making Neural Programming Architectures Generalize via Recursion, ICLR17 / Jonathon Cai, Richard Shin, Dawn Song/ Empirically, neural networks that attempt to learn programs from data have exhibited poor generalizability. Moreover, it has traditionally been difficult to reason about the behavior of these models beyond a certain level of input complexity. In order to address these issues, we propose augmenting neural architectures with a key abstraction: recursion. As an application, we implement recursion in the Neural Programmer-Interpreter framework on four tasks: grade-school addition, bubble sort, topological sort, and quicksort. We demonstrate superior generalizability and interpretability with small amounts of training data. Recursion divides the problem into smaller pieces and drastically reduces the domain of each neural network component, making it tractable to prove guarantees about the overall system's behavior. Our experience suggests that in order for neural architectures to robustly learn program semantics, it is necessary to incorporate a concept like recursion. </sup></sub>



[^2]: <sub><sup>  Learning to Query, Reason, and Answer Questions On Ambiguous Texts, ICLR17/ A key goal of research in conversational systems is to train an interactive agent to help a user with a task. Human conversation, however, is notoriously incomplete, ambiguous, and full of extraneous detail. To operate effectively, the agent must not only understand what was explicitly conveyed but also be able to reason in the presence of missing or unclear information. When unable to resolve ambiguities on its own, the agent must be able to ask the user for the necessary clarifications and incorporate the response in its reasoning. Motivated by this problem we introduce QRAQ ("crack"; Query, Reason, and Answer Questions), a new synthetic domain, in which a User gives an Agent a short story and asks a challenge question. These problems are designed to test the reasoning and interaction capabilities of a learning-based Agent in a setting that requires multiple conversational turns. A good Agent should ask only non-deducible, relevant questions until it has enough information to correctly answer the User's question. We use standard and improved reinforcement learning based memory-network architectures to solve QRAQ problems in the difficult setting where the reward signal only tells the Agent if its final answer to the challenge question is correct or not. To provide an upper-bound to the RL results we also train the same architectures using supervised information that tells the Agent during training which variables to query and the answer to the challenge question. We evaluate our architectures on four QRAQ dataset types, and scale the complexity for each along multiple dimensions. </sup></sub>


[^3]: <sub><sup> Towards Deep Interpretability (MUS-ROVER II): Learning Hierarchical Representations of Tonal Music, ICLR17 / Music theory studies the regularity of patterns in music to capture concepts underlying music styles and composers' decisions. This paper continues the study of building \emph{automatic theorists} (rovers) to learn and represent music concepts that lead to human interpretable knowledge and further lead to materials for educating people. Our previous work took a first step in algorithmic concept learning of tonal music, studying high-level representations (concepts) of symbolic music (scores) and extracting interpretable rules for composition. This paper further studies the representation \emph{hierarchy} through the learning process, and supports \emph{adaptive} 2D memory selection in the resulting language model. This leads to a deeper-level interpretability that expands from individual rules to a dynamic system of rules, making the entire rule learning process more cognitive. The outcome is a new rover, MUS-ROVER \RN{2}, trained on Bach's chorales, which outputs customizable syllabi for learning compositional rules. We demonstrate comparable results to our music pedagogy, while also presenting the differences and variations. In addition, we point out the rover's potential usages in style recognition and synthesis, as well as applications beyond music. </sup></sub>
---
layout: post
title: Reliable Applications V - Understanding2
desc: 2017-W10
tags:
- 3Reliable
categories: 2017Course
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| ChaoJiang |  Toward Deeper Understanding of Neural Networks: The Power of Initialization and a Dual View on Expressivity, NIPS16 | [PDF](https://arxiv.org/abs/1602.05897)| [PDF]({{site.baseurl}}/talks/20171024-Chao.pdf) |
| Rita | Visualizing Deep Neural Network Decisions: Prediction Difference Analysis, ICLR17 | [PDF](https://arxiv.org/abs/1702.04595) | [PDF]({{site.baseurl}}/talks/20171024-Rita.pdf) |
| Xueying | Domain Separation Networks, NIPS16 | [PDF](https://arxiv.org/abs/1608.06019) | [PDF]({{site.baseurl}}/talks/20171024-Xueying.pdf) |
|  | The Robustness of Estimator Composition, NIPS16 | [PDF](https://arxiv.org/abs/1609.01226) |
| Arshdeep | Axiomatic Attribution for Deep Networks, ICML17 | [PDF](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) | [PDF]({{site.baseurl}}/talks/20171031-Arshdeep.pdf) |

---
layout: post
title: Reliable Applications IV - Robustness 
desc: 2017-W9
tags:
- 3Reliable
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| GaoJi | Delving into Transferable Adversarial Examples and Black-box Attacks,ICLR17  [^1]| [pdf](https://arxiv.org/abs/1611.02770) | [PDF]({{site.baseurl}}/talks/20171019-Ji.pdf) |
| Shijia | On Detecting Adversarial Perturbations, ICLR17 [^2] | [pdf](https://arxiv.org/abs/1702.04267) | [PDF]({{site.baseurl}}/talks/20171019-Shijia.pdf) |
| Anant | Parseval Networks: Improving Robustness to Adversarial Examples, ICML17 [^3]| [pdf](https://arxiv.org/abs/1704.08847) | [PDF]({{site.baseurl}}/talks/20171019-Anant.pdf) |
| Bargav | Being Robust (in High Dimensions) Can Be Practical, ICML17 [^4] | [pdf](https://arxiv.org/abs/1703.00893) | [PDF]({{site.baseurl}}/talks/20171019-Bargav.pdf) |



[^1]: <sub><sup> Delving into Transferable Adversarial Examples and Black-box Attacks,ICLR17 / Down Song et al, high cite / An intriguing property of deep neural networks is the existence of adversarial examples, which can transfer among different architectures. These transferable adversarial examples may severely hinder deep neural network-based applications. Previous works mostly study the transferability using small scale datasets. In this work, we are the first to conduct an extensive study of the transferability over large models and a large scale dataset, and we are also the first to study the transferability of targeted adversarial examples with their target labels. We study both non-targeted and targeted adversarial examples, and show that while transferable non-targeted adversarial examples are easy to find, targeted adversarial examples generated using existing approaches almost never transfer with their target labels. Therefore, we propose novel ensemble-based approaches to generating transferable adversarial examples. Using such approaches, we observe a large proportion of targeted adversarial examples that are able to transfer with their target labels for the first time. We also present some geometric studies to help understanding the transferable adversarial examples. Finally, we show that the adversarial examples generated using ensemble-based approaches can successfully attack Clarifai.com, which is a black-box image classification system. </sup></sub>



[^2]: <sub><sup>  On Detecting Adversarial Perturbations, ICLR17 / Machine learning and deep learning in particular has advanced tremendously on perceptual tasks in recent years. However, it remains vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to a human. In this work, we propose to augment deep neural networks with a small "detector" subnetwork which is trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. Our method is orthogonal to prior work on addressing adversarial perturbations, which has mostly focused on making the classification network itself more robust. We show empirically that adversarial perturbations can be detected surprisingly well even though they are quasi-imperceptible to humans. Moreover, while the detectors have been trained to detect only a specific adversary, they generalize to similar and weaker adversaries. In addition, we propose an adversarial attack that fools both the classifier and the detector and a novel training procedure for the detector that counteracts this attack. </sup></sub>


[^3]: <sub><sup> Parseval Networks: Improving Robustness to Adversarial Examples, ICML17  / We introduce Parseval networks, a form of deep neural networks in which the Lipschitz constant of linear, convolutional and aggregation layers is constrained to be smaller than 1. Parseval networks are empirically and theoretically motivated by an analysis of the robustness of the predictions made by deep neural networks when their input is subject to an adversarial perturbation. The most important feature of Parseval networks is to maintain weight matrices of linear and convolutional layers to be (approximately) Parseval tight frames, which are extensions of orthogonal matrices to non-square matrices. We describe how these constraints can be maintained efficiently during SGD. We show that Parseval networks match the state-of-the-art in terms of accuracy on CIFAR-10/100 and Street View House Numbers (SVHN) while being more robust than their vanilla counterpart against adversarial examples. Incidentally, Parseval networks also tend to train faster and make a better usage of the full capacity of the networks. </sup></sub>



[^4]: <sub><sup>  Being Robust (in High Dimensions) Can Be Practical, ICML17/ Robust estimation is much more challenging in high dimensions than it is in one dimension: Most techniques either lead to intractable optimization problems or estimators that can tolerate only a tiny fraction of errors. Recent work in theoretical computer science has shown that, in appropriate distributional models, it is possible to robustly estimate the mean and covariance with polynomial time algorithms that can tolerate a constant fraction of corruptions, independent of the dimension. However, the sample and time complexity of these algorithms is prohibitively large for high-dimensional applications. In this work, we address both of these issues by establishing sample complexity bounds that are optimal, up to logarithmic factors, as well as giving various refinements that allow the algorithms to tolerate a much larger fraction of corruptions. Finally, we show on both synthetic and real data that our algorithms have state-of-the-art performance and suddenly make high-dimensional robust estimation a realistic possibility.  </sup></sub>
---
layout: post
title: Reliable Applications VI - Robustness2
desc: 2017-W10
tags:
- 3Reliable
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Tianlu | Robustness of classifiers: from adversarial to random noise, NIPS16 | [PDF](https://arxiv.org/abs/1608.08967) [^1] | [PDF]({{site.baseurl}}/talks/20171026-Tianlu.pdf) |
| Anant |  Blind Attacks on Machine Learners, [^2] NIPS16 | [PDF](https://papers.nips.cc/paper/6482-blind-attacks-on-machine-learners) | [PDF]({{site.baseurl}}/talks/20171026-Anant.pdf) |
|  | Data Noising as Smoothing in Neural Network Language Models (Ng), ICLR17    [^3]| [pdf](https://arxiv.org/abs/1703.02573) |
|  | The Robustness of Estimator Composition, NIPS16 [^4]| [PDF](https://arxiv.org/abs/1609.01226) |


[^1]: <sub><sup>  The Robustness of Estimator Composition, NIPS16 / We formalize notions of robustness for composite estimators via the notion of a breakdown point. A composite estimator successively applies two (or more) estimators: on data decomposed into disjoint parts, it applies the first estimator on each part, then the second estimator on the outputs of the first estimator. And so on, if the composition is of more than two estimators. Informally, the breakdown point is the minimum fraction of data points which if significantly modified will also significantly modify the output of the estimator, so it is typically desirable to have a large breakdown point. Our main result shows that, under mild conditions on the individual estimators, the breakdown point of the composite estimator is the product of the breakdown points of the individual estimators. We also demonstrate several scenarios, ranging from regression to statistical testing, where this analysis is easy to apply, useful in understanding worst case robustness, and sheds powerful insights onto the associated data analysis. </sup></sub>

[^2]: <sub><sup>  Data Noising as Smoothing in Neural Network Language Models (Ng), ICLR17/ Data noising is an effective technique for regularizing neural network models. While noising is widely adopted in application domains such as vision and speech, commonly used noising primitives have not been developed for discrete sequence-level settings such as language modeling. In this paper, we derive a connection between input noising in neural network language models and smoothing in n-gram models. Using this connection, we draw upon ideas from smoothing to develop effective noising schemes. We demonstrate performance gains when applying the proposed schemes to language modeling and machine translation. Finally, we provide empirical analysis validating the relationship between noising and smoothing. </sup></sub>

[^3]: <sub><sup>  Robustness of classifiers: from adversarial to random noise, NIPS16/ Several recent works have shown that state-of-the-art classifiers are vulnerable to worst-case (i.e., adversarial) perturbations of the datapoints. On the other hand, it has been empirically observed that these same classifiers are relatively robust to random noise. In this paper, we propose to study a semi-random noise regime that generalizes both the random and worst-case noise regimes. We propose the first quantitative analysis of the robustness of nonlinear classifiers in this general noise regime. We establish precise theoretical bounds on the robustness of classifiers in this general regime, which depend on the curvature of the classifier's decision boundary. Our bounds confirm and quantify the empirical observations that classifiers satisfying curvature constraints are robust to random noise. Moreover, we quantify the robustness of classifiers in terms of the subspace dimension in the semi-random noise regime, and show that our bounds remarkably interpolate between the worst-case and random noise regimes. We perform experiments and show that the derived bounds provide very accurate estimates when applied to various state-of-the-art deep neural networks and datasets. This result suggests bounds on the curvature of the classifiers' decision boundaries that we support experimentally, and more generally offers important insights onto the geometry of high dimensional classification problems. </sup></sub>


[^4]: <sub><sup>  Blind Attacks on Machine Learners,  NIPS16/ The importance of studying the robustness of learners to malicious data is well established. While much work has been done establishing both robust estimators and effective data injection attacks when the attacker is omniscient, the ability of an attacker to provably harm learning while having access to little information is largely unstudied. We study the potential of a "blind attacker" to provably limit a learner's performance by data injection attack without observing the learner's training set or any parameter of the distribution from which it is drawn. We provide examples of simple yet effective attacks in two settings: firstly, where an "informed learner" knows the strategy chosen by the attacker, and secondly, where a "blind learner" knows only the proportion of malicious data and some family to which the malicious distribution chosen by the attacker belongs. For each attack, we analyze minimax rates of convergence and establish lower bounds on the learner's minimax risk, exhibiting limits on a learner's ability to learn under data injection attack even when the attacker is "blind". </sup></sub>
---
layout: post
title: Optimization I - Understanding DNN Optimization
desc: 2017-W11
tags:
- 4Optimization
categories: 2017Course
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Ceyer | An overview of gradient optimization algorithms, [^1] | [PDF](https://arxiv.org/abs/1609.04747) | [PDF]({{site.baseurl}}/talks/20171031-Ceyer.pdf) |
| Shijia | Osborne - Probabilistic numerics for deep learning [^2]| [DLSS 2017](https://drive.google.com/file/d/0B2A1tnmq5zQdWHBYOFctNi1KdVU/view) + [Video](http://videolectures.net/deeplearning2017_osborne_probabilistic_numerics/) | [PDF]({{site.baseurl}}/talks/20171031-Shijia.pdf) / [PDF2]({{site.baseurl}}/talks/20171130-Shijia.pdf) |
| Jack | Automated Curriculum Learning for Neural Networks, ICML17 [^3] | [PDF](https://arxiv.org/abs/1704.03003) | [PDF]({{site.baseurl}}/talks/20171031-Jack.pdf) |
| DLSS17 |  Johnson - Automatic Differentiation [^4]| [slide](https://drive.google.com/file/d/0B6NHiPcsmak1ckYxR2hmRGdzdFk/view) + [video](http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/) |



[^1]: <sub><sup>  An overview of gradient optimization algorithms, by S Ruder - ‎2016  Cite as‎: ‎arXiv:1609.04747 / Gradient descent optimization algorithms, while increasingly popular, are often used as black-box optimizers, as practical explanations of their strengths and weaknesses are hard to come by. This article aims to provide the reader with intuitions with regard to the behaviour of different algorithms that will allow her to put them to use. In the course of this overview, we look at different variants of gradient descent, summarize challenges, introduce the most common optimization algorithms, review architectures in a parallel and distributed setting, and investigate additional strategies for optimizing gradient descent. </sup></sub>



[^2]: <sub><sup>  Osborne - Probabilistic numerics for deep learning / http://probabilistic-numerics.org/  Numerical algorithms, such as methods for the numerical solution of integrals and ordinary differential equations, as well as optimization algorithms can be interpreted as estimation rules. They estimate the value of a latent, intractable quantity – the value of an integral, the solution of a differential equation, the location of an extremum – given the result of tractable computations (“observations”, such as function values of the integrand, evaluations of the differential equation, function values of the gradient of an objective). So these methods perform inference, and are accessible to the formal frameworks of probability theory. They are learning machines. Taking this observation seriously, a probabilistic numerical method is a numerical algorithm that takes in a probability distribution over its inputs, and returns a probability distribution over its output. Recent research shows that it is in fact possible to directly identify existing numerical methods, including some real classics, with specific probabilistic models. Interpreting numerical methods as learning algorithms offers various benefits. It can offer insight into the algebraic assumptions inherent in existing methods. As a joint framework for methods developed in separate communities, it allows transfer of knowledge among these areas. But the probabilistic formulation also explicitly provides a richer output than simple convergence bounds. If the probability measure returned by a probabilistic method is well-calibrated, it can be used to monitor, propagate and control the quality of computations. </sup></sub>


[^3]: <sub><sup> Johnson - Automatic Differentiation and more: The simple essence of automatic differentiation /  https://arxiv.org/abs/1804.00746 / Automatic differentiation (AD) in reverse mode (RAD) is a central component of deep learning and other uses of large-scale optimization. Commonly used RAD algorithms such as backpropagation, however, are complex and stateful, hindering deep understanding, improvement, and parallel execution. This paper develops a simple, generalized AD algorithm calculated from a simple, natural specification. The general algorithm is then specialized by varying the representation of derivatives. In particular, applying well-known constructions to a naive representation yields two RAD algorithms that are far simpler than previously known. In contrast to commonly used RAD implementations, the algorithms defined here involve no graphs, tapes, variables, partial derivatives, or mutation. They are inherently parallel-friendly, correct by construction, and usable directly from an existing programming language with no need for new data types or programming style, thanks to use of an AD-agnostic compiler plugin. </sup></sub>


[^4]: <sub><sup> Automated Curriculum Learning for Neural Networks, ICML17 / lex Graves, Marc G. Bellemare, Jacob Menick, Remi Munos, Koray Kavukcuoglu/ We introduce a method for automatically selecting the path, or syllabus, that a neural network follows through a curriculum so as to maximise learning efficiency. A measure of the amount that the network learns from each data sample is provided as a reward signal to a nonstationary multi-armed bandit algorithm, which then determines a stochastic syllabus. We consider a range of signals derived from two distinct indicators of learning progress: rate of increase in prediction accuracy, and rate of increase in network complexity. Experimental results for LSTM networks on three curricula demonstrate that our approach can significantly accelerate learning, in some cases halving the time required to attain a satisfactory performance level. </sup></sub>

---
layout: post
title: Optimization II -  DNN for Optimization
desc: 2017-W11
tags:
- 4Optimization
categories: 2017Course
---



| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| GaoJi | Neural Architecture Search with Reinforcement Learning, ICLR17 [^1] | [PDF](https://openreview.net/pdf?id=r1Ue8Hcxg) | [PDF]({{site.baseurl}}/talks/20171102-Ji.pdf) |
| Ceyer | Learning to learn [^2] | [DLSS17video](http://videolectures.net/deeplearning2017_de_freitas_learning_to_learn/) | [PDF]({{site.baseurl}}/talks/20171102-Ceyer.pdf) |
| Beilun |   Optimization as a Model for Few-Shot Learning, ICLR17 [^3] | [PDF](https://openreview.net/pdf?id=rJY0-Kcll) + [More](https://github.com/songrotek/Meta-Learning-Papers)| [PDF]({{site.baseurl}}/talks/20171102-beilun.pdf) |
| Anant |  Neural Optimizer Search with Reinforcement Learning, ICML17 [^4] |[PDF](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf) | [PDF]({{site.baseurl}}/talks/20171109-Anant.pdf) |

[^1]: <sub><sup>  Neural Optimizer Search with Reinforcement Learning, ICML17 Irwan Bello, Barret Zoph, Vijay Vasudevan, Quoc V. Le / We present an approach to automate the process of discovering optimization methods, with a focus on deep learning architectures. We train a Recurrent Neural Network controller to generate a string in a domain specific language that describes a mathematical update equation based on a list of primitive functions, such as the gradient, running average of the gradient, etc. The controller is trained with Reinforcement Learning to maximize the performance of a model after a few epochs. On CIFAR-10, our method discovers several update rules that are better than many commonly used optimizers, such as Adam, RMSProp, or SGD with and without Momentum on a ConvNet model. We introduce two new optimizers, named PowerSign and AddSign, which we show transfer well and improve training on a variety of different tasks and architectures, including ImageNet classification and Google's neural machine translation system. </sup></sub>



[^2]: <sub><sup>  Neural Architecture Search with Reinforcement Learning, ICLR17 , Barret Zoph, Quoc V. Le / Neural networks are powerful and flexible models that work well for many difficult learning tasks in image, speech and natural language understanding. Despite their success, neural networks are still hard to design. In this paper, we use a recurrent network to generate the model descriptions of neural networks and train this RNN with reinforcement learning to maximize the expected accuracy of the generated architectures on a validation set. On the CIFAR-10 dataset, our method, starting from scratch, can design a novel network architecture that rivals the best human-invented architecture in terms of test set accuracy. Our CIFAR-10 model achieves a test error rate of 3.65, which is 0.09 percent better and 1.05x faster than the previous state-of-the-art model that used a similar architectural scheme. On the Penn Treebank dataset, our model can compose a novel recurrent cell that outperforms the widely-used LSTM cell, and other state-of-the-art baselines. Our cell achieves a test set perplexity of 62.4 on the Penn Treebank, which is 3.6 perplexity better than the previous state-of-the-art model. The cell can also be transferred to the character language modeling task on PTB and achieves a state-of-the-art perplexity of 1.214. </sup></sub>



[^3]: <sub><sup> Learning to learn / DLSS17 / Learning to learn without gradient descent by gradient descent / Yutian Chen, Matthew W Hoffman, Sergio Gómez Colmenarejo, Misha Denil, Timothy P Lillicrap, Matt Botvinick, Nando de Freitas/ We learn recurrent neural network optimizers trained on simple synthetic functions by gradient descent. We show that these learned optimizers exhibit a remarkable degree of transfer in that they can be used to efficiently optimize a broad range of derivative-free black-box functions, including Gaussian process bandits, simple control objectives, global optimization benchmarks and hyper-parameter tuning tasks. Up to the training horizon, the learned optimizers learn to tradeoff exploration and exploitation, and compare favourably with heavily engineered Bayesian optimization packages for hyper-parameter tuning. </sup></sub>



[^4]: <sub><sup> Optimization as a Model for Few-Shot Learning, ICLR17 / achin Ravi, Hugo Larochelle/ Abstract: Though deep neural networks have shown great success in the large data domain, they generally perform poorly on few-shot learning tasks, where a model has to quickly generalize after seeing very few examples from each class. The general belief is that gradient-based optimization in high capacity models requires many iterative steps over many examples to perform well. Here, we propose an LSTM-based meta-learner model to learn the exact optimization algorithm used to train another learner neural network in the few-shot regime. The parametrization of our model allows it to learn appropriate parameter updates specifically for the scenario where a set amount of updates will be made, while also learning a general initialization of the learner network that allows for quick convergence of training. We demonstrate that this meta-learning model is competitive with deep metric-learning techniques for few-shot learning. </sup></sub>---
layout: post
title: Optimization III -   Optimization for DNN
desc: 2017-W12
tags:
- 4Optimization
categories: 2017Course
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| GaoJi  | Forward and Reverse Gradient-Based Hyperparameter Optimization, ICML17  [^1]| [PDF](https://arxiv.org/abs/1703.01785) | [PDF]({{site.baseurl}}/talks/20171107-Ji.pdf) |
| Chaojiang | Adaptive Neural Networks for Efficient Inference, ICML17 [^2] | [PDF](http://proceedings.mlr.press/v70/bolukbasi17a/bolukbasi17a.pdf) | [PDF]({{site.baseurl}}/talks/20171107-Chao.pdf) |
| Bargav | Practical Gauss-Newton Optimisation for Deep Learning, ICML17 [^3]| [PDF](https://arxiv.org/abs/1706.03662) | [PDF]({{site.baseurl}}/talks/20171107-Bargav.pdf) |
| Rita | How to Escape Saddle Points Efficiently,  ICML17 [^4] | [PDF](https://arxiv.org/abs/1703.00887) | [PDF]({{site.baseurl}}/talks/20171107-Rita.pdf) |
|  | Batched High-dimensional Bayesian Optimization via Structural Kernel Learning | [PDF](https://arxiv.org/abs/1703.01973)|



[^1]: <sub><sup> Forward and Reverse Gradient-Based Hyperparameter Optimization, ICML17/ We study two procedures (reverse-mode and forward-mode) for computing the gradient of the validation error with respect to the hyperparameters of any iterative learning algorithm such as stochastic gradient descent. These procedures mirror two methods of computing gradients for recurrent neural networks and have different trade-offs in terms of running time and space requirements. Our formulation of the reverse-mode procedure is linked to previous work by Maclaurin et al. [2015] but does not require reversible dynamics. The forward-mode procedure is suitable for real-time hyperparameter updates, which may significantly speed up hyperparameter optimization on large datasets. We present experiments on data cleaning and on learning task interactions. We also present one large-scale experiment where the use of previous gradient-based methods would be prohibitive. </sup></sub>



[^2]: <sub><sup> Adaptive Neural Networks for Efficient Inference, ICML17 / We present an approach to adaptively utilize deep neural networks in order to reduce the evaluation time on new examples without loss of accuracy. Rather than attempting to redesign or approximate existing networks, we propose two schemes that adaptively utilize networks. We first pose an adaptive network evaluation scheme, where we learn a system to adaptively choose the components of a deep network to be evaluated for each example. By allowing examples correctly classified using early layers of the system to exit, we avoid the computational time associated with full evaluation of the network. We extend this to learn a network selection system that adaptively selects the network to be evaluated for each example. We show that computational time can be dramatically reduced by exploiting the fact that many examples can be correctly classified using relatively efficient networks and that complex, computationally costly networks are only necessary for a small fraction of examples. We pose a global objective for learning an adaptive early exit or network selection policy and solve it by reducing the policy learning problem to a layer-by-layer weighted binary classification problem. Empirically, these approaches yield dramatic reductions in computational cost, with up to a 2.8x speedup on state-of-the-art networks from the ImageNet image recognition challenge with minimal (<1%) loss of top5 accuracy. </sup></sub>


[^3]: <sub><sup>  Practical Gauss-Newton Optimisation for Deep Learning, ICML17 / We present an efficient block-diagonal approximation to the Gauss-Newton matrix for feedforward neural networks. Our resulting algorithm is competitive against state-of-the-art first order optimisation methods, with sometimes significant improvement in optimisation performance. Unlike first-order methods, for which hyperparameter tuning of the optimisation parameters is often a laborious process, our approach can provide good performance even when used with default settings. A side result of our work is that for piecewise linear transfer functions, the network objective function can have no differentiable local maxima, which may partially explain why such transfer functions facilitate effective optimisation. </sup></sub>




[^4]: <sub><sup> How to Escape Saddle Points Efficiently,  ICML17 / Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M. Kakade, Michael I. Jordan/ This paper shows that a perturbed form of gradient descent converges to a second-order stationary point in a number iterations which depends only poly-logarithmically on dimension (i.e., it is almost "dimension-free"). The convergence rate of this procedure matches the well-known convergence rate of gradient descent to first-order stationary points, up to log factors. When all saddle points are non-degenerate, all second-order stationary points are local minima, and our result thus shows that perturbed gradient descent can escape saddle points almost for free. Our results can be directly applied to many machine learning applications, including deep learning. As a particular concrete example of such an application, we show that our results can be used directly to establish sharp global convergence rates for matrix factorization. Our results rely on a novel characterization of the geometry around saddle points, which may be of independent interest to the non-convex optimization community. </sup></sub>


---
layout: post
title: Optimization IV -   change DNN architecture for Optimization 
desc: 2017-W12
tags:
- 4Optimization
categories: 2017Course
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Shijia | Professor Forcing: A New Algorithm for Training Recurrent Networks, [^1] NIPS16 | [PDF](https://arxiv.org/abs/1610.09038) + [Video](http://videolectures.net/deeplearning2016_goyal_new_algorithm/)| [PDF]({{site.baseurl}}/talks/20171109-Shijia.pdf) |
| Beilun+Arshdeep |  Mollifying Networks, Bengio, ICLR17  [^2]| [PDF](https://arxiv.org/abs/1608.04980) | [PDF]({{site.baseurl}}/talks/20171109-Arshdeep.pdf) / [PDF2]({{site.baseurl}}/talks/20171109-BeilunArshdeep.pdf) |


[^1]: <sub><sup> Mollifying Networks, Bengio, ICLR17/ The optimization of deep neural networks can be more challenging than traditional convex optimization problems due to the highly non-convex nature of the loss function, e.g. it can involve pathological landscapes such as saddle-surfaces that can be difficult to escape for algorithms based on simple gradient descent. In this paper, we attack the problem of optimization of highly non-convex neural networks by starting with a smoothed -- or mollified -- objective function that gradually has a more non-convex energy landscape during the training. Our proposition is inspired by the recent studies in continuation methods: similar to curriculum methods, we begin learning an easier (possibly convex) objective function and let it evolve during the training, until it eventually goes back to being the original, difficult to optimize, objective function. The complexity of the mollified networks is controlled by a single hyperparameter which is annealed during the training. We show improvements on various difficult optimization tasks and establish a relationship with recent works on continuation methods for neural networks and mollifiers. </sup></sub>




[^2]: <sub><sup>  Professor Forcing: A New Algorithm for Training Recurrent Networks, NIPS16/ The Teacher Forcing algorithm trains recurrent networks by supplying observed sequence values as inputs during training and using the network's own one-step-ahead predictions to do multi-step sampling. We introduce the Professor Forcing algorithm, which uses adversarial domain adaptation to encourage the dynamics of the recurrent network to be the same when training the network and when sampling from the network over multiple time steps. We apply Professor Forcing to language modeling, vocal synthesis on raw waveforms, handwriting generation, and image generation. Empirically we find that Professor Forcing acts as a regularizer, improving test likelihood on character level Penn Treebank and sequential MNIST. We also find that the model qualitatively improves samples, especially when sampling for a large number of time steps. This is supported by human evaluation of sample quality. Trade-offs between Professor Forcing and Scheduled Sampling are discussed. We produce T-SNEs showing that Professor Forcing successfully makes the dynamics of the network during training and sampling more similar. </sup></sub>

---
layout: post
title: Generative II - Deep Generative Models
desc: 2017-W13
tags:
- 5Generative
categories: 2017Course
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| ChaoJiang | Courville - Generative Models II | [DLSS17Slide](https://drive.google.com/file/d/0B_wzP_JlVFcKQ21udGpTSkh0aVk/view) + [video](http://videolectures.net/deeplearning2017_courville_generative_models/) |  [PDF]({{site.baseurl}}/talks/20171116-Chao.pdf) |
| GaoJi  | Attend, Infer, Repeat: Fast Scene Understanding with Generative Models, NIPS16 [^1]| [PDF](https://arxiv.org/abs/1603.08575) + [talk](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/slides/attend-infer-repeat.pdf)|  [PDF]({{site.baseurl}}/talks/20171114-Ji.pdf) |
| Arshdeep | Composing graphical models with neural networks for structured representations and fast inference, NIPS16 [^2]| [PDF](https://arxiv.org/abs/1603.06277) | [PDF]({{site.baseurl}}/talks/20171114-Arshdeep.pdf) |
|  | Johnson - Graphical Models and Deep Learning | [DLSSSlide](https://drive.google.com/file/d/0B6NHiPcsmak1RmZ3bmtFWUd5bjA/view?usp=drive_web) + [video](http://videolectures.net/deeplearning2017_johnson_graphical_models/)  |
|  | Parallel Multiscale Autoregressive Density Estimation, ICML17 [^3]| [PDF](https://arxiv.org/abs/1703.03664) |
| Beilun | Conditional Image Generation with Pixel CNN Decoders, NIPS16 [^4]| [PDF](https://arxiv.org/abs/1606.05328) | [PDF]({{site.baseurl}}/talks/20171017-beilun.pdf) |
| Shijia | Marrying Graphical Models &	Deep Learning | [DLSS17](http://videolectures.net/site/normal_dl/tag=1129736/deeplearning2017_welling_inference_01.pdf) + [Video](http://videolectures.net/deeplearning2017_welling_inference/)|  [PDF]({{site.baseurl}}/talks/20171121-Shijia.pdf) |





[^1]: <sub><sup>  Attend, Infer, Repeat: Fast Scene Understanding with Generative Models, NIPS16 / Google DeepMind/ We present a framework for efficient inference in structured image models that explicitly reason about objects. We achieve this by performing probabilistic inference using a recurrent neural network that attends to scene elements and processes them one at a time. Crucially, the model itself learns to choose the appropriate number of inference steps. We use this scheme to learn to perform inference in partially specified 2D models (variable-sized variational auto-encoders) and fully specified 3D models (probabilistic renderers). We show that such models learn to identify multiple objects - counting, locating and classifying the elements of a scene -without any supervision, e.g., decomposing 3D images with various numbers of objects in a single forward pass of a neural network at unprecedented speed. We further show that the networks produce accurate inferences when compared to supervised counterparts, and that their structure leads to improved generalization. </sup></sub>



[^2]: <sub><sup>  Composing graphical models with neural networks for structured representations and fast inference, NIPS16 / We propose a general modeling and inference framework that combines the complementary strengths of probabilistic graphical models and deep learning methods. Our model family composes latent graphical models with neural network observation likelihoods. For inference, we use recognition networks to produce local evidence potentials, then combine them with the model distribution using efficient message-passing algorithms. All components are trained simultaneously with a single stochastic variational inference objective. We illustrate this framework by automatically segmenting and categorizing mouse behavior from raw depth video, and demonstrate several other example models. </sup></sub>



[^3]: <sub><sup> Parallel Multiscale Autoregressive Density Estimation, ICML17 / , Nando de Freitas/ PixelCNN achieves state-of-the-art results in density estimation for natural images. Although training is fast, inference is costly, requiring one network evaluation per pixel; O(N) for N pixels. This can be sped up by caching activations, but still involves generating each pixel sequentially. In this work, we propose a parallelized PixelCNN that allows more efficient inference by modeling certain pixel groups as conditionally independent. Our new PixelCNN model achieves competitive density estimation and orders of magnitude speedup - O(log N) sampling instead of O(N) - enabling the practical generation of 512x512 images. We evaluate the model on class-conditional image generation, text-to-image synthesis, and action-conditional video generation, showing that our model achieves the best results among non-pixel-autoregressive density models that allow efficient sampling. </sup></sub>


[^4]: <sub><sup> Conditional Image Generation with Pixel CNN Decoders, NIPS16 / Google DeepMind/ This work explores conditional image generation with a new image density model based on the PixelCNN architecture. The model can be conditioned on any vector, including descriptive labels or tags, or latent embeddings created by other networks. When conditioned on class labels from the ImageNet database, the model is able to generate diverse, realistic scenes representing distinct animals, objects, landscapes and structures. When conditioned on an embedding produced by a convolutional network given a single image of an unseen face, it generates a variety of new portraits of the same person with different facial expressions, poses and lighting conditions. We also show that conditional PixelCNN can serve as a powerful decoder in an image autoencoder. Additionally, the gated convolutional layers in the proposed model improve the log-likelihood of PixelCNN to match the state-of-the-art performance of PixelRNN on ImageNet, with greatly reduced computational cost. </sup></sub>

---
layout: post
title: Generative III -  GAN training 
desc: 2017-W13
tags:
- 5Generative
categories: 2017Course
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Arshdeep  | Generalization and Equilibrium in Generative Adversarial Nets (ICML17) [^1] | [PDF](http://proceedings.mlr.press/v70/arora17a.html) + [video](https://www.youtube.com/watch?v=V7TliSCqOwI) |  [PDF]({{site.baseurl}}/talks/20171116-Arshdeep-1.pdf) |
| Arshdeep  | Mode Regularized Generative Adversarial Networks (ICLR17) [^2]| [PDF](https://arxiv.org/abs/1612.02136)  |  [PDF]({{site.baseurl}}/talks/20171116-Arshdeep-2.pdf) |
| Bargav | Improving Generative Adversarial Networks with Denoising Feature Matching, ICLR17 [^3] | [PDF](https://openreview.net/pdf?id=S1X7nhsxl) |  [PDF]({{site.baseurl}}/talks/20171116-Bargav.pdf) |
| Anant| Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy, ICLR17 [^4]| [PDF](https://arxiv.org/abs/1611.04488) + [code](https://github.com/dougalsutherland/opt-mmd) |  [PDF]({{site.baseurl}}/talks/20171116-Anant.pdf) |



[^1]: <sub><sup> Generalization and Equilibrium in Generative Adversarial Nets (ICML17)/ We show that training of generative adversarial network (GAN) may not have good generalization properties; e.g., training may appear successful but the trained distribution may be far from target distribution in standard metrics. However, generalization does occur for a weaker metric called neural net distance. It is also shown that an approximate pure equilibrium exists in the discriminator/generator game for a special class of generators with natural training objectives when generator capacity and training set sizes are moderate. This existence of equilibrium inspires MIX+GAN protocol, which can be combined with any existing GAN training, and empirically shown to improve some of them. </sup></sub>



[^2]: <sub><sup>  Mode Regularized Generative Adversarial Networks (ICLR17)/ Although Generative Adversarial Networks achieve state-of-the-art results on a variety of generative tasks, they are regarded as highly unstable and prone to miss modes. We argue that these bad behaviors of GANs are due to the very particular functional shape of the trained discriminators in high dimensional spaces, which can easily make training stuck or push probability mass in the wrong direction, towards that of higher concentration than that of the data generating distribution. We introduce several ways of regularizing the objective, which can dramatically stabilize the training of GAN models. We also show that our regularizers can help the fair distribution of probability mass across the modes of the data generating distribution, during the early phases of training and thus providing a unified solution to the missing modes problem. </sup></sub>


[^3]: <sub><sup>  Improving Generative Adversarial Networks with Denoising Feature Matching, ICLR17 / David Warde-Farley, Yoshua Bengio/ Abstract: We propose an augmented training procedure for generative adversarial networks designed to address shortcomings of the original by directing the generator towards probable configurations of abstract discriminator features. We estimate and track the distribution of these features, as computed from data, with a denoising auto-encoder, and use it to propose high-level targets for the generator. We combine this new loss with the original and evaluate the hybrid criterion on the task of unsupervised image synthesis from datasets comprising a diverse set of visual categories, noting a qualitative and quantitative improvement in the objectness'' of the resulting samples. </sup></sub>



[^4]: <sub><sup> Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy, ICLR17 / Dougal J. Sutherland, Hsiao-Yu Tung, Heiko Strathmann, Soumyajit De, Aaditya Ramdas, Alex Smola, Arthur Gretton / We propose a method to optimize the representation and distinguishability of samples from two probability distributions, by maximizing the estimated power of a statistical test based on the maximum mean discrepancy (MMD). This optimized MMD is applied to the setting of unsupervised learning by generative adversarial networks (GAN), in which a model attempts to generate realistic samples, and a discriminator attempts to tell these apart from data samples. In this context, the MMD may be used in two roles: first, as a discriminator, either directly on the samples, or on features of the samples. Second, the MMD can be used to evaluate the performance of a generative model, by testing the model's samples against a reference data set. In the latter role, the optimized MMD is particularly helpful, as it gives an interpretable indication of how the model and data distributions differ, even in cases where individual model samples are not easily distinguished either by eye or by classifier. </sup></sub>---
layout: post
title: RL II - Basic tutorial RLSS17
desc: 2017-W14
tags:
- 6Reinforcement
categories: 2017Course
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Jack | Hasselt - Deep Reinforcement Learning | [RLSS17.pdf](https://drive.google.com/file/d/0BzUSSMdMszk6UE5TbWdZekFXSE0/view?usp=drive_web) + [video](http://videolectures.net/deeplearning2017_van_hasselt_deep_reinforcement/) |  [PDF]({{site.baseurl}}/talks/20171121-Jack.pdf) |
| Tianlu | Roux - RL in the Industry | [RLSS17.pdf](https://drive.google.com/file/d/0BzUSSMdMszk6bEprTUpCaHRrQ28/view) + [video](http://videolectures.net/deeplearning2017_le_roux_recommendation_system/) |  [PDF]({{site.baseurl}}/talks/20171121-Tianlu.pdf) / [PDF-Bandit]({{site.baseurl}}/talks/20171201-Tianlu.pdf) |
| Xueying | Singh - Steps Towards Continual Learning | [pdf](https://drive.google.com/file/d/0BzUSSMdMszk6YVhFUUNLZnZLSWs/view?usp=drive_web) + [video](http://videolectures.net/deeplearning2017_singh_reinforcement_learning/) |  [PDF]({{site.baseurl}}/talks/20171130-Xueying.pdf) |
| GaoJi | Distral: Robust Multitask Reinforcement Learning [^1] | [PDF](https://arxiv.org/pdf/1707.04175.pdf) |  [PDF]({{site.baseurl}}/talks/20171121-Ji.pdf) |







[^1]: <sub><sup>  Distral: Robust Multitask Reinforcement Learning / 2017 Yee Whye Teh, Victor Bapst, Wojciech Marian Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, Razvan Pascanu/ Most deep reinforcement learning algorithms are data inefficient in complex and rich environments, limiting their applicability to many scenarios. One direction for improving data efficiency is multitask learning with shared neural network parameters, where efficiency may be improved through transfer across related tasks. In practice, however, this is not usually observed, because gradients from different tasks can interfere negatively, making learning unstable and sometimes even less data efficient. Another issue is the different reward schemes between tasks, which can easily lead to one task dominating the learning of a shared model. We propose a new approach for joint training of multiple tasks, which we refer to as Distral (Distill & transfer learning). Instead of sharing parameters between the different workers, we propose to share a "distilled" policy that captures common behaviour across tasks. Each worker is trained to solve its own task while constrained to stay close to the shared policy, while the shared policy is trained by distillation to be the centroid of all task policies. Both aspects of the learning process are derived by optimizing a joint objective function. We show that our approach supports efficient transfer on complex 3D environments, outperforming several related methods. Moreover, the proposed learning process is more robust and more stable---attributes that are critical in deep reinforcement learning. </sup></sub>


---
layout: post
title: RL III - Basic tutorial RLSS17 (2)
desc: 2017-W14
tags:
- 6Reinforcement
categories: 2017Course
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Anant | The Predictron: End-to-End Learning and Planning, ICLR17 [^1]| [PDF](https://openreview.net/pdf?id=BkJsCIcgl) |  [PDF]({{site.baseurl}}/talks/20171130-Anant.pdf) |
| ChaoJiang | Szepesvari - Theory of RL  [^2]| [RLSS.pdf](https://drive.google.com/file/d/0BzUSSMdMszk6U194Ym5jSnZQbGM/view?usp=drive_web) + [Video](http://videolectures.net/deeplearning2017_szepesvari_theory_of_rl/)|  [PDF]({{site.baseurl}}/talks/20171130-Chao.pdf) |
| GaoJi  | Mastering the game of Go without human knowledge / Nature 2017 [^3] | [PDF](https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html)  |  [PDF]({{site.baseurl}}/talks/20171130-Ji.pdf) |
| | Thomas - Safe Reinforcement Learning | [RLSS17.pdf](https://drive.google.com/file/d/0BzUSSMdMszk6TDRMRGRaM0dBcHM/view?usp=drive_web) + [video](http://videolectures.net/deeplearning2017_thomas_safe_rl/) |
|  | Sutton - Temporal-Difference Learning | [RLSS17.pdf](https://drive.google.com/file/d/0BzUSSMdMszk6VE9kMkY2SzQzSW8/view?usp=drive_web) + [Video](http://videolectures.net/deeplearning2017_sutton_td_learning/) |






[^1]: <sub><sup>  Temporal-Difference Learning basics / https://en.wikipedia.org/wiki/Temporal_difference_learning /  Temporal difference (TD) learning refers to a class of model-free reinforcement learning methods which learn by bootstrapping from the current estimate of the value function. These methods sample from the environment, like Monte Carlo methods, and perform updates based on current estimates, like dynamic programming methods. While Monte Carlo methods only adjust their estimates once the final outcome is known, TD methods adjust predictions to match later, more accurate, predictions about the future before the final outcome is known. </sup></sub>


[^2]: <sub><sup>  The Predictron: End-to-End Learning and Planning, ICLR17 / David Silver/ One of the key challenges of artificial intelligence is to learn models that are effective in the context of planning. In this document we introduce the predictron architecture. The predictron consists of a fully abstract model, represented by a Markov reward process, that can be rolled forward multiple "imagined" planning steps. Each forward pass of the predictron accumulates internal rewards and values over multiple planning depths. The predictron is trained end-to-end so as to make these accumulated values accurately approximate the true value function. We applied the predictron to procedurally generated random mazes and a simulator for the game of pool. The predictron yielded significantly more accurate predictions than conventional deep neural network architectures. </sup></sub>



[^3]: <sub><sup>  Mastering the game of Go without human knowledge / Nature 2017 Google DeepMind / A long-standing goal of artificial intelligence is an algorithm that learns, tabula rasa, superhuman proficiency in challenging domains. Recently, AlphaGo became the first program to defeat a world champion in the game of Go. The tree search in AlphaGo evaluated positions and selected moves using deep neural networks. These neural networks were trained by supervised learning from human expert moves, and by reinforcement learning from self-play. Here we introduce an algorithm based solely on reinforcement learning, without human data, guidance or domain knowledge beyond game rules. AlphaGo becomes its own teacher: a neural network is trained to predict AlphaGo’s own move selections and also the winner of AlphaGo’s games. This neural network improves the strength of the tree search, resulting in higher quality move selection and stronger self-play in the next iteration. Starting tabula rasa, our new program AlphaGo Zero achieved superhuman performance, winning 100–0 against the previously published, champion-defeating AlphaGo. </sup></sub>
---
layout: post
title: RL IV - RL with varying structures
desc: 2017-W15
tags:
- 6Reinforcement
categories: 2017Course
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Ceyer | Reinforcement Learning with Unsupervised Auxiliary Tasks, ICLR17 [^1]| [PDF](https://arxiv.org/pdf/1611.05397.pdf) |  [PDF]({{site.baseurl}}/talks/20171201-Ceyer.pdf) |
| Beilun  | Why is Posterior Sampling Better than Optimism for Reinforcement Learning? Ian Osband, Benjamin Van Roy [^2]| [PDF](https://arxiv.org/abs/1607.00215) | [PDF]({{site.baseurl}}/talks/20171201-Beilun.pdf) |
| Ji | Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction, ICML17 [^3] | [PDF](http://proceedings.mlr.press/v70/sun17d.html) |  [PDF]({{site.baseurl}}/talks/20171201-Ji.pdf) |
| Xueying | End-to-End Differentiable Adversarial Imitation Learning, ICML17  [^4]| [PDF](http://proceedings.mlr.press/v70/baram17a.html) |  [PDF]({{site.baseurl}}/talks/20171201-Xueying.pdf) |
|  | Deep Value Networks Learn to Evaluate and Iteratively Refine Structured Outputs, ICML17 | [PDF](http://proceedings.mlr.press/v70/gygli17a.html) |
|  | FeUdal Networks for Hierarchical Reinforcement Learning, ICML17 [^5] | [PDF](https://arxiv.org/pdf/1703.01161.pdf) |



[^1]: <sub><sup>  Reinforcement Learning with Unsupervised Auxiliary Tasks, ICLR17 / Max Jaderberg, Volodymyr Mnih, Wojciech Marian Czarnecki, Tom Schaul, Joel Z Leibo, David Silver, Koray Kavukcuoglu/ Deep reinforcement learning agents have achieved state-of-the-art results by directly maximising cumulative reward. However, environments contain a much wider variety of possible training signals. In this paper, we introduce an agent that also maximises many other pseudo-reward functions simultaneously by reinforcement learning. All of these tasks share a common representation that, like unsupervised learning, continues to develop in the absence of extrinsic rewards. We also introduce a novel mechanism for focusing this representation upon extrinsic rewards, so that learning can rapidly adapt to the most relevant aspects of the actual task. Our agent significantly outperforms the previous state-of-the-art on Atari, averaging 880\% expert human performance, and a challenging suite of first-person, three-dimensional \emph{Labyrinth} tasks leading to a mean speedup in learning of 10× and averaging 87\% expert human performance on Labyrinth. </sup></sub>


[^2]: <sub><sup>  Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction, ICML17/ Researchers have demonstrated state-of-the-art performance in sequential decision making problems (e.g., robotics control, sequential prediction) with deep neural network models. One often has access to near-optimal oracles that achieve good performance on the task during training. We demonstrate that AggreVaTeD --- a policy gradient extension of the Imitation Learning (IL) approach of (Ross & Bagnell, 2014) --- can leverage such an oracle to achieve faster and better solutions with less training data than a less-informed Reinforcement Learning (RL) technique. Using both feedforward and recurrent neural network predictors, we present stochastic gradient procedures on a sequential prediction task, dependency-parsing from raw image data, as well as on various high dimensional robotics control problems. We also provide a comprehensive theoretical study of IL that demonstrates we can expect up to exponentially lower sample complexity for learning with AggreVaTeD than with RL algorithms, which backs our empirical findings. Our results and theory indicate that the proposed approach can achieve superior performance with respect to the oracle when the demonstrator is sub-optimal. </sup></sub>




[^3]: <sub><sup>  End-to-End Differentiable Adversarial Imitation Learning, ICML17/ Generative Adversarial Networks (GANs) have been successfully applied to the problem of policy imitation in a model-free setup. However, the computation graph of GANs, that include a stochastic policy as the generative model, is no longer differentiable end-to-end, which requires the use of high-variance gradient estimation. In this paper, we introduce the Model-based Generative Adversarial Imitation Learning (MGAIL) algorithm. We show how to use a forward model to make the computation fully differentiable, which enables training policies using the exact gradient of the discriminator. The resulting algorithm trains competent policies using relatively fewer expert samples and interactions with the environment. We test it on both discrete and continuous action domains and report results that surpass the state-of-the-art. </sup></sub>



[^4]: <sub><sup>   Deep Value Networks Learn to Evaluate and Iteratively Refine Structured Outputs, ICML17/ We approach structured output prediction by optimizing a deep value network (DVN) to precisely estimate the task loss on different output configurations for a given input. Once the model is trained, we perform inference by gradient descent on the continuous relaxations of the output variables to find outputs with promising scores from the value network. When applied to image segmentation, the value network takes an image and a segmentation mask as inputs and predicts a scalar estimating the intersection over union between the input and ground truth masks. For multi-label classification, the DVN's objective is to correctly predict the F1 score for any potential label configuration. The DVN framework achieves the state-of-the-art results on multi-label prediction and image segmentation benchmarks. </sup></sub>



[^5]: <sub><sup> FeUdal Networks for Hierarchical Reinforcement Learning, ICML17 / Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David Silver, Koray Kavukcuoglu/ We introduce FeUdal Networks (FuNs): a novel architecture for hierarchical reinforcement learning. Our approach is inspired by the feudal reinforcement learning proposal of Dayan and Hinton, and gains power and efficacy by decoupling end-to-end learning across multiple levels -- allowing it to utilise different resolutions of time. Our framework employs a Manager module and a Worker module. The Manager operates at a lower temporal resolution and sets abstract goals which are conveyed to and enacted by the Worker. The Worker generates primitive actions at every tick of the environment. The decoupled structure of FuN conveys several benefits -- in addition to facilitating very long timescale credit assignment it also encourages the emergence of sub-policies associated with different goals set by the Manager. These properties allow FuN to dramatically outperform a strong baseline agent on tasks that involve long-term credit assignment or memorisation. We demonstrate the performance of our proposed system on a range of tasks from the ATARI suite and also from a 3D DeepMind Lab environment. </sup></sub>---
layout: post
title: Application18- Property of DeepNN Models and Discrete tasks 
desc: 2018-team
categories: 2018Reads
tags:
- 3Reliable
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Bill | Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation [^1]| [PDF](https://arxiv.org/abs/1609.08144) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.02.09_GoogleNMT.pdf) | 
| Bill |  Measuring the tendency of CNNs to Learn Surface Statistical Regularities Jason Jo, Yoshua Bengio | [PDF](https://arxiv.org/abs/1711.11561) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.02.16_CNNRegularity.pdf) | 
| Bill | Generating Sentences by Editing Prototypes, Kelvin Guu, Tatsunori B. Hashimoto, Yonatan Oren, Percy Liang [^2]  | [PDF](https://arxiv.org/abs/1709.08878) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.01.25_PrototypeEdit.pdf) | 
| Bill | On the importance of single directions for generalization, Ari S. Morcos, David G.T. Barrett, Neil C. Rabinowitz, Matthew Botvinick  | [PDF](https://arxiv.org/abs/1803.06959) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.04.28_SingleDirections.pdf) | 




[^1]: <sub><sup>  Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation/  Neural Machine Translation (NMT) is an end-to-end learning approach for automated translation, with the potential to overcome many of the weaknesses of conventional phrase-based translation systems. Unfortunately, NMT systems are known to be computationally expensive both in training and in translation inference. Also, most NMT systems have difficulty with rare words. These issues have hindered NMT's use in practical deployments and services, where both accuracy and speed are essential. In this work, we present GNMT, Google's Neural Machine Translation system, which attempts to address many of these issues. Our model consists of a deep LSTM network with 8 encoder and 8 decoder layers using attention and residual connections. To improve parallelism and therefore decrease training time, our attention mechanism connects the bottom layer of the decoder to the top layer of the encoder. To accelerate the final translation speed, we employ low-precision arithmetic during inference computations. To improve handling of rare words, we divide words into a limited set of common sub-word units ("wordpieces") for both input and output. This method provides a good balance between the flexibility of "character"-delimited models and the efficiency of "word"-delimited models, naturally handles translation of rare words, and ultimately improves the overall accuracy of the system. Our beam search technique employs a length-normalization procedure and uses a coverage penalty, which encourages generation of an output sentence that is most likely to cover all the words in the source sentence. On the WMT'14 English-to-French and English-to-German benchmarks, GNMT achieves competitive results to state-of-the-art. Using a human side-by-side evaluation on a set of isolated simple sentences, it reduces translation errors by an average of 60% compared to Google's phrase-based production system. </sup></sub>




[^2]: <sub><sup> Generating Sentences by Editing Prototypes, Kelvin Guu, Tatsunori B. Hashimoto, Yonatan Oren, Percy Liang / We propose a new generative model of sentences that first samples a prototype sentence from the training corpus and then edits it into a new sentence. Compared to traditional models that generate from scratch either left-to-right or by first sampling a latent sentence vector, our prototype-then-edit model improves perplexity on language modeling and generates higher quality outputs according to human evaluation. Furthermore, the model gives rise to a latent edit vector that captures interpretable semantics such as sentence similarity and sentence-level analogies. </sup></sub>
---
layout: post
title: Survey18- My Survey Talk at UVA HMI Seminar - A quick and rough overview of DNN
desc: 2018-me
tags:
- 0Survey
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Dr. Qi | A quick and rough survey of Deep-Neural-Networks |  |  [PDF]({{site.baseurl}}/talks/201802-QI-HMI-DeepOverview.pdf) |

---
layout: post
title: Reliable18- Adversarial Attacks and DNN 
desc: 2018-team
tags:
- 3Reliable
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Bill |  Intriguing Properties of Adversarial Examples, Ekin D. Cubuk, Barret Zoph, Samuel S. Schoenholz, Quoc V. Le [^1] | [PDF](https://arxiv.org/abs/1711.02846) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.02.23_AdversarialProperties.pdf) | 
| Bill |  Adversarial Spheres [^2] | [PDF](https://arxiv.org/abs/1801.02774) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.03.16_AdversarialSpheres.pdf) | 
| Bill |  Adversarial Transformation Networks: Learning to Generate Adversarial Examples, Shumeet Baluja, Ian Fischer [^3]| [PDF](https://arxiv.org/abs/1703.09387) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.03.16_TransformNetwork.pdf) | 
| Bill |  Thermometer encoding: one hot way to resist adversarial examples [^4]| [PDF](https://openreview.net/pdf?id=S18Su--CW) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.03.23_TemperatureEncoding.pdf) | 
| | Adversarial Logit Pairing , Harini Kannan, Alexey Kurakin, Ian Goodfellow [^5] | [PDF](https://arxiv.org/abs/1803.06373) | | 

[^1]: <sub><sup>  Intriguing Properties of Adversarial Examples, Ekin D. Cubuk, Barret Zoph, Samuel S. Schoenholz, Quoc V. Le / It is becoming increasingly clear that many machine learning classifiers are vulnerable to adversarial examples. In attempting to explain the origin of adversarial examples, previous studies have typically focused on the fact that neural networks operate on high dimensional data, they overfit, or they are too linear. Here we argue that the origin of adversarial examples is primarily due to an inherent uncertainty that neural networks have about their predictions. We show that the functional form of this uncertainty is independent of architecture, dataset, and training protocol; and depends only on the statistics of the logit differences of the network, which do not change significantly during training. This leads to adversarial error having a universal scaling, as a power-law, with respect to the size of the adversarial perturbation. We show that this universality holds for a broad range of datasets (MNIST, CIFAR10, ImageNet, and random data), models (including state-of-the-art deep networks, linear models, adversarially trained networks, and networks trained on randomly shuffled labels), and attacks (FGSM, step l.l., PGD). Motivated by these results, we study the effects of reducing prediction entropy on adversarial robustness. Finally, we study the effect of network architectures on adversarial sensitivity. To do this, we use neural architecture search with reinforcement learning to find adversarially robust architectures on CIFAR10. Our resulting architecture is more robust to white \emph{and} black box attacks compared to previous attempts </sup></sub>

[^2]: <sub><sup>  Adversarial Spheres / Ian Goodfellow/ State of the art computer vision models have been shown to be vulnerable to small adversarial perturbations of the input. In other words, most images in the data distribution are both correctly classified by the model and are very close to a visually similar misclassified image. Despite substantial research interest, the cause of the phenomenon is still poorly understood and remains unsolved. We hypothesize that this counter intuitive behavior is a naturally occurring result of the high dimensional geometry of the data manifold. As a first step towards exploring this hypothesis, we study a simple synthetic dataset of classifying between two concentric high dimensional spheres. For this dataset we show a fundamental tradeoff between the amount of test error and the average distance to nearest error. In particular, we prove that any model which misclassifies a small constant fraction of a sphere will be vulnerable to adversarial perturbations of size O(1/d‾‾√). Surprisingly, when we train several different architectures on this dataset, all of their error sets naturally approach this theoretical bound. As a result of the theory, the vulnerability of neural networks to small adversarial perturbations is a logical consequence of the amount of test error observed. We hope that our theoretical analysis of this very simple case will point the way forward to explore how the geometry of complex real-world data sets leads to adversarial examples. </sup></sub>



[^3]: <sub><sup> Adversarial Transformation Networks: Learning to Generate Adversarial Examples, Shumeet Baluja, Ian Fischer/ With the rapidly increasing popularity of deep neural networks for image recognition tasks, a parallel interest in generating adversarial examples to attack the trained models has arisen. To date, these approaches have involved either directly computing gradients with respect to the image pixels or directly solving an optimization on the image pixels. We generalize this pursuit in a novel direction: can a separate network be trained to efficiently attack another fully trained network? We demonstrate that it is possible, and that the generated attacks yield startling insights into the weaknesses of the target network. We call such a network an Adversarial Transformation Network (ATN). ATNs transform any input into an adversarial attack on the target network, while being minimally perturbing to the original inputs and the target network’s outputs. Further, we show that ATNs are capable of not only causing the target network to make an error, but can be constructed to explicitly control the type of misclassification made. We demonstrate ATNs on both simple MNIST digit classifiers and state-of-the-art ImageNet classifiers deployed by Google, Inc.: Inception ResNet-v2.</sup></sub>



[^4]: <sub><sup> Thermometer encoding: one hot way to resist adversarial examples / It is well known that for neural networks, it is possible to construct inputs which are misclassified by the network yet indistinguishable from true data points, known as adversarial examples. We propose a simple modification to standard neural network architectures, \emph{thermometer encoding}, which significantly increases the robustness of the network to adversarial examples. We demonstrate this robustness with experiments on the MNIST, CIFAR-10, CIFAR-100, and SVHN datasets, and show that models with thermometer-encoded inputs consistently have higher accuracy on adversarial examples, while also maintaining the same accuracy on non-adversarial examples and training more quickly. </sup></sub>



[^5]: <sub><sup>  Adversarial Logit Pairing , Harini Kannan, Alexey Kurakin, Ian Goodfellow (Submitted on 16 Mar 2018)/ In this paper, we develop improved techniques for defending against adversarial examples at scale. First, we implement the state of the art version of adversarial training at unprecedented scale on ImageNet and investigate whether it remains effective in this setting - an important open scientific question (Athalye et al., 2018). Next, we introduce enhanced defenses using a technique we call logit pairing, a method that encourages logits for pairs of examples to be similar. When applied to clean examples and their adversarial counterparts, logit pairing improves accuracy on adversarial examples over vanilla adversarial training; we also find that logit pairing on clean examples only is competitive with adversarial training in terms of accuracy on two datasets. Finally, we show that adversarial logit pairing achieves the state of the art defense on ImageNet against PGD white box attacks, with an accuracy improvement from 1.5% to 27.9%. Adversarial logit pairing also successfully damages the current state of the art defense against black box attacks on ImageNet (Tramer et al., 2018), dropping its accuracy from 66.6% to 47.1%. With this new accuracy drop, adversarial logit pairing ties with Tramer et al.(2018) for the state of the art on black box attacks on ImageNet. </sup></sub>

---
layout: post
title: Generative18 -Generative Adversarial Network (classified)
desc: 2018-team
tags:
- 5Generative
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| BrandonLiu | Summary of Recent Generative Adversarial Networks (Classified)  |  |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un18/Liu18May-GANSummary.pdf) | 
| Jack |  Generating and designing DNA with deep generative models, Nathan Killoran, Leo J. Lee, Andrew Delong, David Duvenaud, Brendan J. Frey | [PDF](https://arxiv.org/abs/1712.06148) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Jack/20180218_GeneratingDNA.pdf) | 
| GaoJi |  More about basics of GAN |  |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/JIGAN.pdf) | 
|  | McGan: Mean and Covariance Feature Matching GAN, PMLR 70:2527-2535 | [PDF](https://arxiv.org/abs/1702.08398) |
|  | Wasserstein GAN, ICML17 | [PDF](https://arxiv.org/abs/1701.07875) |
|  | Geometrical Insights for Implicit Generative Modeling, L Bottou, M Arjovsky, D Lopez-Paz, M Oquab  | [PDF](https://arxiv.org/pdf/1712.07822.pdf) |



[^1]: <sub><sup>  McGan: Mean and Covariance Feature Matching GAN, ICML17, PMLR 70:2527-2535/ We introduce new families of Integral Probability Metrics (IPM) for training Generative Adversarial Networks (GAN). Our IPMs are based on matching statistics of distributions embedded in a finite dimensional feature space. Mean and covariance feature matching IPMs allow for stable training of GANs, which we will call McGan. McGan minimizes a meaningful loss between distributions. </sup></sub>



[^2]: <sub><sup>  Wasserstein GAN, ICML17/ We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to different distances between distributions. </sup></sub>




[^3]: <sub><sup>  Generating and designing DNA with deep generative models, Nathan Killoran, Leo J. Lee, Andrew Delong, David Duvenaud, Brendan J. Frey / 2017 / partial / We propose generative neural network methods to generate DNA sequences and tune them to have desired properties. We present three approaches: creating synthetic DNA sequences using a generative adversarial network; a DNA-based variant of the activation maximization ("deep dream") design method; and a joint procedure which combines these two approaches together. We show that these tools capture important structures of the data and, when applied to designing probes for protein binding microarrays, allow us to generate new sequences whose properties are estimated to be superior to those found in the training data. We believe that these results open the door for applying deep generative models to advance genomics research. </sup></sub>



[^4]: <sub><sup> Geometrical Insights for Implicit Generative Modeling/ Learning algorithms for implicit generative models can optimize a variety of criteria that measure how the data distribution differs from the implicit model distribution, including the Wasserstein distance, the Energy distance, and the Maximum Mean Discrepancy criterion. A careful look at the geometries induced by these distances on the space of probability measures reveals interesting differences. In particular, we can establish surprising approximate global convergence guarantees for the 1-Wasserstein distance,even when the parametric generator has a nonconvex parametrization. </sup></sub>
---
layout: post
title: Structures18-  More Attentions 
desc: 2018-team
tags:
- 2Structures
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Arshdeep |  Show, Attend and Tell: Neural Image Caption Generation with Visual Attention [^1] | [PDF](https://arxiv.org/abs/1502.03044) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/ATTENTION-06222018-hardattention.pdf) | 
| Arshdeep |  Latent Alignment and Variational Attention [^2]| [PDF](https://arxiv.org/abs/1807.03756) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/ATTENTION-08172018-VariationalAttention.pdf) | 
| Arshdeep | Modularity Matters: Learning Invariant Relational Reasoning Tasks, Jason Jo, Vikas Verma, Yoshua Bengio [^3]| [PDF](https://arxiv.org/abs/1806.06765) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/DEEP-06292018-InvariantRelationalReasoning.pdf) | 


[^1]: <sub><sup>  Show, Attend and Tell: Neural Image Caption Generation with Visual Attention / Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio/ Inspired by recent work in machine translation and object detection, we introduce an attention based model that automatically learns to describe the content of images. We describe how we can train this model in a deterministic manner using standard backpropagation techniques and stochastically by maximizing a variational lower bound. We also show through visualization how the model is able to automatically learn to fix its gaze on salient objects while generating the corresponding words in the output sequence. We validate the use of attention with state-of-the-art performance on three benchmark datasets: Flickr8k, Flickr30k and MS COCO. </sup></sub>


[^2]: <sub><sup>  Latent Alignment and Variational Attention / NIPS2018 / Yuntian Deng, Yoon Kim, Justin Chiu, Demi Guo, Alexander M. Rush/ Neural attention has become central to many state-of-the-art models in natural language processing and related domains. Attention networks are an easy-to-train and effective method for softly simulating alignment; however, the approach does not marginalize over latent alignments in a probabilistic sense. This property makes it difficult to compare attention to other alignment approaches, to compose it with probabilistic models, and to perform posterior inference conditioned on observed data. A related latent approach, hard attention, fixes these issues, but is generally harder to train and less accurate. This work considers variational attention networks, alternatives to soft and hard attention for learning latent variable alignment models, with tighter approximation bounds based on amortized variational inference. We further propose methods for reducing the variance of gradients to make these approaches computationally feasible. Experiments show that for machine translation and visual question answering, inefficient exact latent variable models outperform standard neural attention, but these gains go away when using hard attention based training. On the other hand, variational attention retains most of the performance gain but with training speed comparable to neural attention. </sup></sub>

[^3]: <sub><sup> Modularity Matters: Learning Invariant Relational Reasoning Tasks, Jason Jo, Vikas Verma, Yoshua Bengio  / ICML18 /  We focus on two supervised visual reasoning tasks whose labels encode a semantic relational rule between two or more objects in an image: the MNIST Parity task and the colorized Pentomino task. The objects in the images undergo random translation, scaling, rotation and coloring transformations. Thus these tasks involve invariant relational reasoning. We report uneven performance of various deep CNN models on these two tasks. For the MNIST Parity task, we report that the VGG19 model soundly outperforms a family of ResNet models. Moreover, the family of ResNet models exhibits a general sensitivity to random initialization for the MNIST Parity task. For the colorized Pentomino task, now both the VGG19 and ResNet models exhibit sluggish optimization and very poor test generalization, hovering around 30% test error. The CNN we tested all learn hierarchies of fully distributed features and thus encode the distributed representation prior. We are motivated by a hypothesis from cognitive neuroscience which posits that the human visual cortex is modularized, and this allows the visual cortex to learn higher order invariances. To this end, we consider a modularized variant of the ResNet model, referred to as a Residual Mixture Network (ResMixNet) which employs a mixture-of-experts architecture to interleave distributed representations with more specialized, modular representations. We show that very shallow ResMixNets are capable of learning each of the two tasks well, attaining less than 2% and 1% test error on the MNIST Parity and the colorized Pentomino tasks respectively. Most importantly, the ResMixNet models are extremely parameter efficient: generalizing better than various non-modular CNNs that have over 10x the number of parameters. These experimental results support the hypothesis that modularity is a robust prior for learning invariant relational reasoning. </sup></sub>---
layout: post
title: Reliable18- Adversarial Attacks and DNN and More
desc: 2018-team
tags:
- 3Reliable
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Bill |  Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples | [PDF](https://arxiv.org/abs/1803.01128) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.03.30_Seq2Sick.pdf) |
| Bill |  Adversarial Examples for Evaluating Reading Comprehension Systems, Robin Jia, Percy Liang | [PDF](https://arxiv.org/abs/1707.07328) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.04.06_ReadingComp.pdf) | 
| Bill |  Certified Defenses against Adversarial Examples, Aditi Raghunathan, Jacob Steinhardt, Percy Liang | [PDF](https://arxiv.org/abs/1801.09344) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.04.14_CertifiedDefenses.pdf) | 
| Bill |  Provably Minimally-Distorted Adversarial Examples, Nicholas Carlini, Guy Katz, Clark Barrett, David L. Dill | [PDF](https://arxiv.org/pdf/1709.10207.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Bill/18.04.21_MinimumDistort.pdf) | 

---
layout: post
title: Application18- A few DNN for Question Answering
desc: 2018-team
tags:
- 2Structures
categories: 2018Reads
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Derrick |  GloVe: Global Vectors for Word Representation | [PDF](https://nlp.stanford.edu/pubs/glove.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un18/Blakely-GloVe.pdf) | 
| Derrick | PARL.AI: A unified platform for sharing, training and evaluating dialog models across many tasks.  | [URL](http://www.parl.ai/) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un18/Blakely-ParlAI.pdf) | 
| Derrick | scalable nearest neighbor algorithms for high dimensional data (PAMI14) [^1] | [PDF](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_pami2014.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un18/Blakely-ScalableKNN.pdf) | 
| Derrick |   StarSpace: Embed All The Things! | [PDF](https://arxiv.org/abs/1709.03856) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un18/Blakely-StarSpace.pdf) | 
|  Derrick | Weaver: Deep Co-Encoding of Questions and Documents for Machine Reading, Martin Raison, Pierre-Emmanuel Mazaré, Rajarshi Das, Antoine Bordes  | [PDF](https://arxiv.org/abs/1804.10490) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Un18/Blakely-Weaver.pdf) | 



[^1]: <sub><sup> Salable nearest neighbor algorithms for high dimensional data (PAMI14) / https://www.ncbi.nlm.nih.gov/pubmed/26353063 / For many computer vision and machine learning problems, large training sets are key for good performance. However, the most computationally expensive part of many computer vision and machine learning algorithms consists of finding nearest neighbor matches to high dimensional vectors that represent the training data. We propose new algorithms for approximate nearest neighbor matching and evaluate and compare them with previous algorithms. For matching high dimensional features, we find two algorithms to be the most efficient: the randomized k-d forest and a new algorithm proposed in this paper, the priority search k-means tree. We also propose a new algorithm for matching binary features by searching multiple hierarchical clustering trees and show it outperforms methods typically used in the literature. We show that the optimal nearest neighbor algorithm and its parameters depend on the data set characteristics and describe an automated configuration procedure for finding the best algorithm to search a particular data set. In order to scale to very large data sets that would otherwise not fit in the memory of a single machine, we propose a distributed nearest neighbor matching framework that can be used with any of the algorithms described in the paper. All this research has been released as an open source library called fast library for approximate nearest neighbors (FLANN), which has been incorporated into OpenCV and is now one of the most popular libraries for nearest neighbor matching. </sup></sub>

---
layout: post
title: Reliable18- Testing and Verifying DNNs
desc: 2018-team
tags:
- 3Reliable
- 6Reinforcement
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| GaoJi |  Deep Reinforcement Fuzzing, Konstantin Böttinger, Patrice Godefroid, Rishabh Singh | [PDF](https://arxiv.org/abs/1801.04589) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/Ji_RLTesting.pdf) | 
| GaoJi | Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks, Guy Katz, Clark Barrett, David Dill, Kyle Julian, Mykel Kochenderfer  | [PDF](https://arxiv.org/abs/1702.01135) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/Ji_VerifyML.pdf) | 
| GaoJi | DeepTest: Automated Testing of Deep-Neural-Network-driven Autonomous Cars, Yuchi Tian, Kexin Pei, Suman Jana, Baishakhi Ray  | [PDF](https://arxiv.org/abs/1708.08559) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/Ji-DeepTest.pdf) | 
| GaoJi | A few Recent (2018) papers on Black-box Adversarial Attacks, like Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors [^1]| [PDF](https://arxiv.org/abs/1807.07978) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/JIBlack-boxAE.pdf) | 
| GaoJi | A few Recent papers of Adversarial Attacks on reinforcement learning, like Adversarial Attacks on Neural Network Policies (Sandy Huang, Nicolas Papernot, Ian Goodfellow, Yan Duan, Pieter Abbeel)| [PDF](https://arxiv.org/abs/1702.02284) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/JI-RLAE.pdf) | 


[^1]: <sub><sup> Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors / ICLR19/ We study the problem of generating adversarial examples in a black-box setting in which only loss-oracle access to a model is available. We introduce a framework that conceptually unifies much of the existing work on black-box attacks, and demonstrate that the current state-of-the-art methods are optimal in a natural sense. Despite this optimality, we show how to improve black-box attacks by bringing a new element into the problem: gradient priors. We give a bandit optimization-based algorithm that allows us to seamlessly integrate any such priors, and we explicitly identify and incorporate two examples. The resulting methods use two to four times fewer queries and fail two to five times less than the current state-of-the-art. The code for reproducing our work is available at https://git.io/fAjOJ. </sup></sub>
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


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Arshdeep |  DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning.  | [PDF](https://www.ncbi.nlm.nih.gov/pubmed/30035154) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BRAIN-07242018-BlogPostsDeepLesionChestXRay.pdf) | 
| Arshdeep | Solving the RNA design problem with reinforcement learning, PLOSCB  [^1] | [PDF](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006176) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-06292018-RNAReinforcement.pdf) | 
| Arshdeep | Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk [^2] | [PDF](https://www.nature.com/articles/s41588-018-0160-6) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-07272018-AbinitioPrediction-SeqtoGeneExp.pdf) | 
| Arshdeep | Towards Gene Expression Convolutions using Gene Interaction Graphs, Francis Dutil, Joseph Paul Cohen, Martin Weiss, Georgy Derevyanko, Yoshua Bengio [^3] | [PDF](https://arxiv.org/abs/1806.06975) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-06292018-GeneInterCNN.pdf) | 
|  Brandon| Kipoi: Accelerating the Community Exchange and Reuse of Predictive Models for Genomics | [PDF](http://kipoi.org/docs/) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Brandon18.9.14Kipoi.pdf) | 






[^3]: <sub><sup> Towards Gene Expression Convolutions using Gene Interaction Graphs, Francis Dutil, Joseph Paul Cohen, Martin Weiss, Georgy Derevyanko, Yoshua Bengio/ We study the challenges of applying deep learning to gene expression data. We find experimentally that there exists non-linear signal in the data, however is it not discovered automatically given the noise and low numbers of samples used in most research. We discuss how gene interaction graphs (same pathway, protein-protein, co-expression, or research paper text association) can be used to impose a bias on a deep model similar to the spatial bias imposed by convolutions on an image. We explore the usage of Graph Convolutional Neural Networks coupled with dropout and gene embeddings to utilize the graph information. We find this approach provides an advantage for particular tasks in a low data regime but is very dependent on the quality of the graph used. We conclude that more work should be done in this direction. We design experiments that show why existing methods fail to capture signal that is present in the data when features are added which clearly isolates the problem that needs to be addressed. </sup></sub>






[^1]: <sub><sup> Solving the RNA design problem with reinforcement learning, PLOSCB/ We use reinforcement learning to train an agent for computational RNA design: given a target secondary structure, design a sequence that folds to that structure in silico. Our agent uses a novel graph convolutional architecture allowing a single model to be applied to arbitrary target structures of any length. After training it on randomly generated targets, we test it on the Eterna100 benchmark and find it outperforms all previous algorithms. Analysis of its solutions shows it has successfully learned some advanced strategies identified by players of the game Eterna, allowing it to solve some very difficult structures. On the other hand, it has failed to learn other strategies, possibly because they were not required for the targets in the training set. This suggests the possibility that future improvements to the training protocol may yield further gains in performance. Author summary: Designing RNA sequences that fold to desired structures is an important problem in bioengineering. We have applied recent advances in machine learning to address this problem. The computer learns without any human input, using only trial and error to figure out how to design RNA. It quickly discovers powerful strategies that let it solve many difficult design problems. When tested on a challenging benchmark, it outperforms all previous algorithms. We analyze its solutions and identify some of the strategies it has learned, as well as other important strategies it has failed to learn. This suggests possible approaches to further improving its performance. This work reflects a paradigm shift taking place in computer science, which has the potential to transform computational biology. Instead of relying on experts to design algorithms by hand, computers can use artificial intelligence to learn their own algorithms directly. The resulting methods often work better than the ones designed by humans. </sup></sub>



[^2]: <sub><sup>  Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk / Nature Geneticsvolume 50, pages1171–1179 (2018)/ Key challenges for human genetics, precision medicine and evolutionary biology include deciphering the regulatory code of gene expression and understanding the transcriptional effects of genome variation. However, this is extremely difficult because of the enormous scale of the noncoding mutation space. We developed a deep learning–based framework, ExPecto, that can accurately predict, ab initio from a DNA sequence, the tissue-specific transcriptional effects of mutations, including those that are rare or that have not been observed. We prioritized causal variants within disease- or trait-associated loci from all publicly available genome-wide association studies and experimentally validated predictions for four immune-related diseases. By exploiting the scalability of ExPecto, we characterized the regulatory mutation space for human RNA polymerase II–transcribed genes by in silico saturation mutagenesis and profiled > 140 million promoter-proximal mutations. This enables probing of evolutionary constraints on gene expression and ab initio prediction of mutation disease effects, making ExPecto an end-to-end computational framework for the in silico prediction of expression and disease risk. </sup></sub>---
layout: post
title: Generative18 -A few more DNN Generative Models
desc: 2018-team
tags:
- 5Generative
categories: 2018Reads
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
|  Arshdeep| The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables, Chris J. Maddison, Andriy Mnih, Yee Whye Teh [^1] | [PDF](https://arxiv.org/abs/1611.00712) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/DEEP-07162018-Gumbel-Softmax.pdf) | 
| Arshdeep | Feedback GAN (FBGAN) for DNA: a Novel Feedback-Loop Architecture for Optimizing Protein Functions [^2] | [PDF](https://arxiv.org/abs/1804.01694) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/GAN-07132018-FBGAN.pdf) | 
| GaoJi | Summary Of Several Autoencoder models  | [PDF]() |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/JiAutoencoderNNs.pdf) | 
| GaoJi | Latent Constraints: Learning to Generate Conditionally from Unconditional Generative Models, Jesse Engel, Matthew Hoffman, Adam Roberts [^3] | [PDF](https://arxiv.org/abs/1711.05772) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/JIConditionalGEN.pdf) | 
| GaoJi |  Summary of A Few Recent Papers about Discrete Generative models, SeqGAN, MaskGAN, BEGAN, BoundaryGAN| [PDF]() |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji/JIDiscreteGAN.pdf) | 
| Arshdeep |  Semi-Amortized Variational Autoencoders, Yoon Kim, Sam Wiseman, Andrew C. Miller, David Sontag, Alexander M. Rush [^4] | [PDF](https://arxiv.org/abs/1802.02550) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/DEEP-07172018-semi-amoritzed-VAE.pdf) | 
| Arshdeep | Synthesizing Programs for Images using Reinforced Adversarial Learning, Yaroslav Ganin, Tejas Kulkarni, Igor Babuschkin, S.M. Ali Eslami, Oriol Vinyals [^5] | [PDF](https://arxiv.org/abs/1804.01118) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/DEEP-07312018-ImageSythesisProgramming.pdf) | 


[^1]: <sub><sup> Synthesizing Programs for Images using Reinforced Adversarial Learning, Yaroslav Ganin, Tejas Kulkarni, Igor Babuschkin, S.M. Ali Eslami, Oriol Vinyals / ICML18/ Advances in deep generative networks have led to impressive results in recent years. Nevertheless, such models can often waste their capacity on the minutiae of datasets, presumably due to weak inductive biases in their decoders. This is where graphics engines may come in handy since they abstract away low-level details and represent images as high-level programs. Current methods that combine deep learning and renderers are limited by hand-crafted likelihood or distance functions, a need for large amounts of supervision, or difficulties in scaling their inference algorithms to richer datasets. To mitigate these issues, we present SPIRAL, an adversarially trained agent that generates a program which is executed by a graphics engine to interpret and sample images. The goal of this agent is to fool a discriminator network that distinguishes between real and rendered data, trained with a distributed reinforcement learning setup without any supervision. A surprising finding is that using the discriminator's output as a reward signal is the key to allow the agent to make meaningful progress at matching the desired output rendering. To the best of our knowledge, this is the first demonstration of an end-to-end, unsupervised and adversarial inverse graphics agent on challenging real world (MNIST, Omniglot, CelebA) and synthetic 3D datasets. </sup></sub>


[^2]: <sub><sup>  Semi-Amortized Variational Autoencoders, Yoon Kim, Sam Wiseman, Andrew C. Miller, David Sontag, Alexander M. Rush / ICML 2018/ Amortized variational inference (AVI) replaces instance-specific local inference with a global inference network. While AVI has enabled efficient training of deep generative models such as variational autoencoders (VAE), recent empirical work suggests that inference networks can produce suboptimal variational parameters. We propose a hybrid approach, to use AVI to initialize the variational parameters and run stochastic variational inference (SVI) to refine them. Crucially, the local SVI procedure is itself differentiable, so the inference network and generative model can be trained end-to-end with gradient-based optimization. This semi-amortized approach enables the use of rich generative models without experiencing the posterior-collapse phenomenon common in training VAEs for problems like text generation. Experiments show this approach outperforms strong autoregressive and variational baselines on standard text and image datasets.


[^3]: <sub><sup>  Feedback GAN (FBGAN) for DNA: a Novel Feedback-Loop Architecture for Optimizing Protein Functions / Anvita Gupta, James Zou (arxiv Submitted on 5 Apr 2018) / Generative Adversarial Networks (GANs) represent an attractive and novel approach to generate realistic data, such as genes, proteins, or drugs, in synthetic biology. Here, we apply GANs to generate synthetic DNA sequences encoding for proteins of variable length. We propose a novel feedback-loop architecture, called Feedback GAN (FBGAN), to optimize the synthetic gene sequences for desired properties using an external function analyzer. The proposed architecture also has the advantage that the analyzer need not be differentiable. We apply the feedback-loop mechanism to two examples: 1) generating synthetic genes coding for antimicrobial peptides, and 2) optimizing synthetic genes for the secondary structure of their resulting peptides. A suite of metrics demonstrate that the GAN generated proteins have desirable biophysical properties. The FBGAN architecture can also be used to optimize GAN-generated datapoints for useful properties in domains beyond genomics. </sup></sub>



[^4]: <sub><sup> The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables, Chris J. Maddison, Andriy Mnih, Yee Whye Teh (2016)/ The reparameterization trick enables optimizing large scale stochastic computation graphs via gradient descent. The essence of the trick is to refactor each stochastic node into a differentiable function of its parameters and a random variable with fixed distribution. After refactoring, the gradients of the loss propagated by the chain rule through the graph are low variance unbiased estimators of the gradients of the expected loss. While many continuous random variables have such reparameterizations, discrete random variables lack useful reparameterizations due to the discontinuous nature of discrete states. In this work we introduce Concrete random variables---continuous relaxations of discrete random variables. The Concrete distribution is a new family of distributions with closed form densities and a simple reparameterization. Whenever a discrete stochastic node of a computation graph can be refactored into a one-hot bit representation that is treated continuously, Concrete stochastic nodes can be used with automatic differentiation to produce low-variance biased gradients of objectives (including objectives that depend on the log-probability of latent stochastic nodes) on the corresponding discrete graph. We demonstrate the effectiveness of Concrete relaxations on density estimation and structured prediction tasks using neural networks. </sup></sub>


[^5]: <sub><sup> Latent Constraints: Learning to Generate Conditionally from Unconditional Generative Models, Jesse Engel, Matthew Hoffman, Adam Roberts , arxiv 2017/ Deep generative neural networks have proven effective at both conditional and unconditional modeling of complex data distributions. Conditional generation enables interactive control, but creating new controls often requires expensive retraining. In this paper, we develop a method to condition generation without retraining the model. By post-hoc learning latent constraints, value functions that identify regions in latent space that generate outputs with desired attributes, we can conditionally sample from these regions with gradient-based optimization or amortized actor functions. Combining attribute constraints with a universal "realism" constraint, which enforces similarity to the data distribution, we generate realistic conditional images from an unconditional variational autoencoder. Further, using gradient-based optimization, we demonstrate identity-preserving transformations that make the minimal adjustment in latent space to modify the attributes of an image. Finally, with discrete sequences of musical notes, we demonstrate zero-shot conditional generation, learning latent constraints in the absence of labeled data or a differentiable reward function. Code with dedicated cloud instance has been made publicly available. </sup></sub>---
layout: post
title: Survey18- My Tutorial Talk at ACM BCB18 - Interpretable Deep Learning for Genomics
desc: 2018-me
tags:
- 8BioApplications
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Dr. Qi | Making Deep Learning Understandable for Analyzing Sequential Data about Gene Regulation |         |  [PDF]({{site.baseurl}}/MoreTalksTeam/20180229-deepBio-BCBtutorial.pdf) |

> #### Attend and Predict: Understanding Gene Regulation by Selective Attention on Chromatin, NIPS2017 / Ritambhara Singh, Jack Lanchantin, Arshdeep Sekhon, Yanjun Qi
>> The past decade has seen a revolution in genomic technologies that enable a flood of genome-wide profiling of chromatin marks. Recent literature tried to understand gene regulation by predicting gene expression from large-scale chromatin measurements. Two fundamental challenges exist for such learning tasks: (1) genome-wide chromatin signals are spatially structured, high-dimensional and highly modular; and (2) the core aim is to understand what are the relevant factors and how they work together? Previous studies either failed to model complex dependencies among input signals or relied on separate feature analysis to explain the decisions. This paper presents an attention-based deep learning approach; we call AttentiveChrome, that uses a unified architecture to model and to interpret dependencies among chromatin factors for controlling gene regulation. AttentiveChrome uses a hierarchy of multiple Long short-term memory (LSTM) modules to encode the input signals and to model how various chromatin marks cooperate automatically. AttentiveChrome trains two levels of attention jointly with the target prediction, enabling it to attend differentially to relevant marks and to locate important positions per mark. We evaluate the model across 56 different cell types (tasks) in human. Not only is the proposed architecture more accurate, but its attention scores also provide a better interpretation than state-of-the-art feature visualization methods such as saliency map. 
Code and data are shared at[www.deepchrome.org](http://deepchrome.org/) ---
layout: post
title: Structures18- DNN for Relations
desc: 2018-team
tags:
- 2Structures
- 7Graphs
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
|  Arshdeep| Relational inductive biases, deep learning, and graph networks  | [PDF](https://arxiv.org/abs/1806.01261) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh18-relationBias.pdf) | 
|  Arshdeep|  Discriminative Embeddings of Latent Variable Models for Structured Data | [PDF](https://arxiv.org/abs/1603.05629) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh18-structure2vec.pdf) | 
|  Jack| Deep Graph Infomax | [PDF](https://openreview.net/forum?id=rklz9iAcKQ) |  [PDF]({{site.baseurl}}/MoreTalksTeam/20181009-Jack-DeepGraphInfomax.pdf) | 
|  Jack| FastXML: A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning | [PDF](http://manikvarma.org/pubs/prabhu14.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/20181018-Jack-FastXML.pdf) | 
|  Chao| Maximizing Subset Accuracy with Recurrent Neural Networks in Multi-label Classification | [PDF](http://www.ke.tu-darmstadt.de/bibtex/publications/show/3017) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Chao18-SubsetMLC.pdf) | 


---
layout: post
title: Reliable18- Understand DNNs 
desc: 2018-team
tags:
- 3Reliable
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
|  Jack| A Unified Approach to Interpreting Model Predictions | [PDF](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) |  [PDF]({{site.baseurl}}/MoreTalksTeam/20181026-Jack-UnifiedApproachInterpretingModelPredictions.pdf) |
|  Jack| "Why Should I Trust You?": Explaining the Predictions of Any Classifier | [PDF](https://arxiv.org/abs/1602.04938) |  [PDF]({{site.baseurl}}/MoreTalksTeam/20181026-Jack-WhyShouldITrustYou.pdf) | 
|  Jack| Visual Feature Attribution using Wasserstein GANs | [PDF](https://arxiv.org/abs/1711.08998) |  [PDF]({{site.baseurl}}/MoreTalksTeam/20181030-Jack-VisualFeatureAttributionusingWassersteinGANs.pdf) | 
|  Jack| GAN Dissection: Visualizing and Understanding Generative Adversarial Networks | [PDF](https://gandissect.csail.mit.edu/) |  [PDF]({{site.baseurl}}/MoreTalksTeam/20181130-Jack-VisualizingandUnderstandingGANs.pdf) | 
|  GaoJi| Recent Interpretable machine learning papers  | [PDF](https://people.csail.mit.edu/beenkim/papers/BeenK_FinaleDV_ICML2017_tutorial.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Ji-f18-interpretable_machinelearning.pdf) | 
| Jennifer | The Building Blocks of Interpretability |   [PDF](https://distill.pub/2018/building-blocks/) | [PDF]({{site.baseurl}}/MoreTalksTeam/Jennifer18-BuildingBlocksInterpretability.pdf)  | 
---
layout: post
title: Application18- DNNs in a Few Bio CRISPR Tasks
desc: 2018-team
tags:
- 8BioApplications
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Arshdeep |  deepCRISPR: optimized CRISPR guide RNA design by deep learning , Genome Biology 2018| [PDF](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1459-4) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-07162018-DeepCRISPR.pdf) | 
|  Arshdeep| The CRISPR tool kit for genome editing and beyond, Mazhar Adli  | [PDF](https://www.nature.com/articles/s41467-018-04252-2) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh/BIO-07062018-CRISPR-Review.pdf) | 
|  Eric| Intro of Genetic Engineering  | [PDF](https://www.nature.com/articles/s41467-018-04252-2) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Eric2018_10_12-19_Genetic_eng_intro.pdf) | 
|  Eric| Prediction of off-target activities for the end-to-end design of CRISPR guide RNAs  | [PDF](https://www.nature.com/articles/s41551-017-0178-6) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Eric18-11_12-CRISPR_Off_target_Listgarten.pdf) | 
---
layout: post
title: Application18- Graph DNN in a Few Bio Tasks
desc: 2018-team
tags:
- 7Graphs
- 8BioApplications
categories: 2018Reads
---

| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
|  Eric| Modeling polypharmacy side effects with graph convolutional networks  |        [PDF](https://arxiv.org/abs/1802.00543) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Eric1_26_2019-2_1_2019-GNNdrug_polypharmacy.pdf) | 
|  Eric| Protein Interface Prediction using Graph Convolutional Networks | [PDF](https://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Eric2_4-2_82019-GNNpr_protein_interface_3DNN.pdf) | 

---
layout: post
title: Structure18- DNNs Varying Structures
desc: 2018-team
tags:
- 2Structures
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
|  Arshdeep| Learning Transferable Architectures for Scalable Image Recognition  | [PDF](https://arxiv.org/abs/1707.07012) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh18-NAStransferable.pdf) | 
| Arshdeep | FractalNet: Ultra-Deep Neural Networks without Residuals | [PDF](https://arxiv.org/abs/1605.07648) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh18-FractalNet.pdf) | 

---
layout: post
title: Generate18- Deep Generative Models for Graphs
desc: 2018-team
tags:
- 5Generative
- 7Graphs
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
|  Arshdeep| Constrained Graph Variational Autoencoders for Molecule Design  | [PDF](https://arxiv.org/abs/1805.09076) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh18-CVAEgraph.pdf) | 
| Arshdeep | Learning Deep Generative Models of Graphs | [PDF](https://arxiv.org/abs/1803.03324) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh18-DGenerateGraph.pdf) | 
|  Arshdeep|  Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation | [PDF](https://arxiv.org/abs/1806.02473) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Arsh18-GCPNgraphGen.pdf) | 
|  Jack | Generating and designing DNA with deep generative models  | [PDF](https://arxiv.org/abs/1712.06148) |  [PDF]({{site.baseurl}}/MoreTalksTeam/20181118-Jack-GeneratingAndDesigningDNA.pdf) | 


---
layout: post
title: Reliable18- Adversarial Attacks and DNN 
desc: 2018-team
tags:
- 3Reliable
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Bill | Adversarial Examples that Fool both Computer Vision and Time-Limited Humans | [PDF](https://arxiv.org/abs/1802.08195) |  [PDF]({{site.baseurl}}/MoreTalksTeam/bill18.08.28_FoolComputerHumans.pdf) | 
| Bill | Adversarial Attacks Against Medical Deep Learning Systems | [PDF](https://arxiv.org/abs/1804.05296) |  [PDF]({{site.baseurl}}/MoreTalksTeam/bill18.09.14_MedicalDL.pdf) | 
| Bill |  TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing | [PDF](https://arxiv.org/abs/1807.10875) |  [PDF]({{site.baseurl}}/MoreTalksTeam/bill18.09.14_TensorFuzz.pdf) | 
| Bill |  Distilling the Knowledge in a Neural Network | [PDF](https://arxiv.org/abs/1503.02531) |  [PDF]({{site.baseurl}}/MoreTalksTeam/bill18.10.05_Distillation.pdf) | 
| Bill |  Defensive Distillation is Not Robust to Adversarial Examples | [PDF](https://arxiv.org/abs/1607.04311) |  [PDF]({{site.baseurl}}/MoreTalksTeam/bill18.10.26_NotRobustDistillation.pdf) | 
| Bill | Adversarial Logit Pairing , Harini Kannan, Alexey Kurakin, Ian Goodfellow  | [PDF](https://arxiv.org/abs/1803.06373) | [PDF]({{site.baseurl}}/MoreTalksTeam/bill18.10.26_AdversarialLogitPairing.pdf) | 

---
layout: post
title: Reliable18- Adversarial Attacks and DNN 
desc: 2018-team
tags:
- 3Reliable
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Jennifer |  Adversarial Attacks Against Medical Deep Learning Systems | [PDF](https://arxiv.org/abs/1804.05296)  |  [PDF]({{site.baseurl}}/MoreTalksTeam/Jennifer18-AdversarialAttacksAgainstMedicalDeepLearningSystems.pdf) | 
| Jennifer |  Adversarial-Playground: A Visualization Suite Showing How Adversarial Examples Fool Deep Learning | [PDF](https://arxiv.org/abs/1708.00807) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Jennifer18-AdversarialPlayground.pdf) | 
| Jennifer | Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers | [PDF](https://arxiv.org/abs/1801.04354) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Jennifer18-Black-boxDeepWordbug.pdf) | 
| Jennifer |  CleverHans | [PDF](https://cleverhans.readthedocs.io/en/latest/) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Jennifer18-CleverHans.pdf) | 
| Ji | Ji-f18-New papers about adversarial attack | | [PDF]({{site.baseurl}}/MoreTalksTeam/Ji-f18-NewAEs.pdf) | 

---
layout: post
title: Application18- DNN for MedQA 
desc: 2018-team
tags:
- 2Structures
- 8BioApplications
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
| Bill | Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning | [PDF](https://arxiv.org/abs/1709.00103) |  [PDF]({{site.baseurl}}/MoreTalksTeam/bill19.01.18_seq2sql.pdf) | 
| Chao | Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis (I) | [PDF](https://arxiv.org/abs/1706.03446) |  [PDF]({{site.baseurl}}/MoreTalksTeam/chao18-DeepEHR1.pdf) | 
| Chao | Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis (II) | [PDF](https://arxiv.org/abs/1706.03446) |  [PDF]({{site.baseurl}}/MoreTalksTeam/chao18-DeepEHR2.pdf) | 
| Derrick | Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis (III) | [PDF](https://arxiv.org/abs/1706.03446) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Derrick18-DeepEHRPresentation.pdf) | 
| Chao | Reading Wikipedia to Answer Open-Domain Questions | [PDF](https://arxiv.org/abs/1704.00051) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Chao18-WikiQA.pdf) | 
| Jennifer | Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text | [PDF](https://arxiv.org/abs/1809.00782) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Jennifer18-OpenDomainQuestionAnsweringEarlyFusionKBText.pdf) | 
---
layout: post
title: Generate18- Deep Generative Models for discrete
desc: 2018-team
tags:
- 5Generative
- 7Graphs
categories: 2018Reads
---


| Presenter | Papers | Paper URL| Our Presentation |
| -----: | ---------------------------: | :----- | :----- |
|  Tkach| Boundary-Seeking Generative Adversarial Networks  | [PDF](https://arxiv.org/abs/1702.08431) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Tkach-BGAN-201809.pdf) | 
| Tkach | Maximum-Likelihood Augmented Discrete Generative Adversarial Networks | [PDF](https://arxiv.org/abs/1702.07983) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Tkach-MAliGAN-201901.pdf) | 
|  Tkach |  Generating	Sentences from a Continuous	Space | [PDF](https://arxiv.org/abs/1511.06349) |  [PDF]({{site.baseurl}}/MoreTalksTeam/Tkach-RNNLMandVAE-1810.pdf) | 

---
layout: post
title: GNN Basics I - Deep Learning Advances on Graphs 
desc: 2019-W1
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---




| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Basics |  Large-scale Graph Representation Learning  by Jure Leskovec Stanford University  |  [URL](http://www.ipam.ucla.edu/abstract/?tid=14555&pcode=DLT2018) + [PDF](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf) |  | Ryan [Pdf]() | 
| Basics  | Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering by Xavier Bresson Nanyang Technological University, Singapore   |  [URL](http://www.ipam.ucla.edu/abstract/?tid=14506&pcode=DLT2018) + [PDF](http://helper.ipam.ucla.edu/publications/dlt2018/dlt2018_14506.pdf) |  | Ryan [Pdf]() | 


---
layout: post
title: GNN Basics II - Deep Learning Advances on Graphs 
desc: 2019-W2
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---

| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Basics| Gated Graph Sequence Neural Networks by Microsoft Research  |  [URL](https://www.youtube.com/watch?v=cWIeTMklzNg) + [PDF](https://arxiv.org/abs/1511.05493) |  | Faizan [Pdf]() |
| Basics | Geometric Deep Learning (simple introduction video) |  [URL](https://www.youtube.com/watch?v=D3fnGG7cdjY) |  | Faizan [Pdf]() | 
| Basics | DeepWalk - Turning Graphs into Features via Network Embeddings  |  [URL](https://www.youtube.com/watch?v=aZNtHJwfIVg) + [PDF](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)|  | Faizan [Pdf]() | 
---
layout: post
title: GNN Basics III - Deep Learning Advances on Graphs 
desc: 2019-W3
categories: 2019sCourse
tags:
- 7Graphs
- 0Survey
gnnType: basics
---


| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Basics |  A Comprehensive Survey on Graph Neural Networks/ Graph Neural Networks: A Review of Methods and Applications   |   [Pdf](https://arxiv.org/pdf/1901.00596.pdf) | Jack [Pdf]() |  | 
| Basics |   Spectral Networks and Locally Connected Networks on Graphs [^1] | [Pdf](https://arxiv.org/abs/1312.6203) | GaoJi [Pdf]() | Bill [Pdf]() | 


[^1]: <sub><sup> Some Relevant Notes from [URL](https://mathoverflow.net/questions/231987/why-decompose-a-function-with-eigenvectors-of-laplace-operator). On periodic domain, people always use Fourier basis, which eigenvectors of Laplace operator. On sphere, people use spherical harmonics, which also are eigenvectors of Laplace operator. In applied science, people decompose functions on a graph using eigenvectors of graph laplacian. Why are these basis preferred? The exponentials used in Fourier series are eigenvalues of shifts, and thus of any operator commuting with shifts, not just Laplacian. Similarly, spherical harmonics carry irreducible representations of 𝑆𝑂(3) and so they are eigenfunctions of any rotationally invariant operator. If the underlying space has symmetries, it's no wonder that a basis respecting those symmetries has some nice properties. </sup></sub>---
layout: post
title: GNN Basics and Applications   
desc: 2019-W4
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---

| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Bio |   Protein Interface Prediction using Graph Convolutional Networks   | [Pdf](https://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf) | Eli [Pdf]() | Jack [Pdf]() | 
| QA |   Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text     | [Pdf](https://arxiv.org/abs/1809.00782) | Bill [Pdf]() | GaoJi [Pdf]() | 
| Understand |   Faithful and Customizable Explanations of Black Box Models    | [Pdf](http://www.aies-conference.com/wp-content/papers/main/AIES-19_paper_143.pdf) | Derrick [Pdf]() |  | 
| Basics |   Graph mining: Laws, generators, and algorithms   | [Pdf](https://dl.acm.org/citation.cfm?id=1132954) | Arshdeep [Pdf]() | | 

---
layout: post
title: GNN Extensions I 
desc: 2019-W5
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---

| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
|  Geometric |   Spherical CNNs   | [Pdf](https://arxiv.org/abs/1801.10130) | Fuwen [Pdf]() | Arshdeep [Pdf]() | 
| Program |  Deep Program Reidentification: A Graph Neural Network Solution | [Pdf](https://arxiv.org/abs/1812.04064) | Weilin [Pdf]() | Jack [Pdf]() | 
| Generate |  Maximum-Likelihood Augmented Discrete Generative Adversarial Networks  | [PDF](https://arxiv.org/abs/1702.07983) | Tkach [PDF]({{site.baseurl}}/MoreTalksTeam/Tkach-MAliGAN-201901.pdf) |  GaoJi [Pdf]() |
| Edge |  Loss-aware Binarization of Deep Networks, ICLR17 | [PDF](https://arxiv.org/abs/1611.01600)   | Ryan [Pdf]() | Derrick [Pdf]() | 
| Robust |   KDD’18 Adversarial Attacks on Neural Networks for Graph Data  | [Pdf](https://www.kdd.org/kdd2018/accepted-papers/view/adversarial-attacks-on-neural-networks-for-graph-data) | Faizan [Pdf]() | GaoJi [Pdf]()|
---
layout: post
title: GNN Extension II   
desc: 2019-W6
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---

| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Basics |  Semi-Supervised Classification with Graph Convolutional Networks + Deep Graph Infomax | [Pdf](https://arxiv.org/abs/1609.02907) | Jack [Pdf]() | Eli [Pdf]() | 
| Understand |  A causal framework for explaining the predictions of black-box sequence-to-sequence models, EMNLP17     | [Pdf](https://arxiv.org/abs/1707.01943) | GaoJi [Pdf]() | Bill [Pdf]() | 
| Generate |  Graphical Generative Adversarial Networks  |  [PDF](https://arxiv.org/abs/1804.03429)  |  Arshdeep [Pdf]() | Tkach [Pdf]() | 
|  QA |   A Comparison of Current Graph Database Models   | [Pdf](http://www.renzoangles.net/files/gdm2012.pdf) + [PDF2](https://users.dcc.uchile.cl/~cgutierr/papers/surveyGDB.pdf) | Bill [Pdf]() |  | 

---
layout: post
title: GNN Applications 
desc: 2019-W7
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---

| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
|  Scalable |  Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis | [Pdf](https://arxiv.org/abs/1802.09941) | Derrick [Pdf]() |  | 
| Edge | DeepX: A Software Accelerator for Low-Power Deep Learning Inference on Mobile Devices  | [Pdf](https://ix.cs.uoregon.edu/~jiao/papers/ipsn16.pdf) | Eamon [Pdf]() | Derrick [Pdf]() | 
| Program | Heterogeneous Graph Neural Networks for Malicious Account Detection  | [Pdf](https://dl.acm.org/citation.cfm?id=3272010) | Weilin [Pdf]() | Jack [Pdf]() | 
---
layout: post
title: GNN  Applications 
desc: 2019-W8
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---


| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Robust | Adversarial Attacks on Graph Structured Data  | [Pdf](https://arxiv.org/abs/1806.02371) | Faizan [Pdf]() | GaoJi [Pdf]() | 
| Geometric | All Graphs Lead to Rome: Learning Geometric and Cycle-Consistent Representations with Graph Convolutional Networks | [Pdf](https://arxiv.org/abs/1611.08097) | Fuwen [Pdf]() |  | 
| Generate |  Junction Tree Variational Autoencoder for Molecular Graph Generation  | [Pdf](https://arxiv.org/abs/1802.04364) | Tkach [Pdf]() | Arshdeep [Pdf]() | 
|  Bio |  KDEEP: Protein–Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks, 2018  |  [Pdf](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00650) | Eli [Pdf]() | Jack [Pdf]() | 
---
layout: post
title: GNN more generate  
desc: 2019-W9
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---

| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Basics |  FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling | [Pdf](https://arxiv.org/abs/1801.10247) | Ryan [Pdf]() | Arshdeep [Pdf]() | 
| Basics | Deep Learning of Graph Matching, + Graph Edit Distance Computation via Graph Neural Networks | [PDF](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zanfir_Deep_Learning_of_CVPR_2018_paper.pdf)+ [PDF](http://robotics.stanford.edu/~quocle/CaeCheLeSmo07.pdf)+ [PDF](https://arxiv.org/pdf/1808.05689.pdf) | Jack [Pdf]() |  | 
| Understand |  How Powerful are Graph Neural Networks? / Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning | [Pdf](https://arxiv.org/abs/1810.00826) + [Pdf](https://arxiv.org/abs/1801.07606) | GaoJi [Pdf]() | Jack [Pdf]() | 
|  QA |  Generative Question Answering: Learning to Answer the Whole Question, Mike Lewis, Angela Fan    | [Pdf](https://openreview.net/forum?id=Bkx0RjA9tX) | Bill [Pdf]() | GaoJi [Pdf]() | 



---
layout: post
title: GNN   
desc: 2019-W10
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---


| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Generate |  GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models, ICML2018  |  [PDF](https://arxiv.org/abs/1802.08773)  |  Arshdeep [Pdf]() | Tkach [Pdf]() | 
|  Scalable |  DNN Dataflow Choice Is Overrated | [PDF](https://arxiv.org/pdf/1809.04070.pdf) | Derrick [Pdf]() |  | 
| Generate |  Towards Variational Generation of Small Graphs  | [Pdf](https://arxiv.org/abs/1802.03480) | Tkach [Pdf]() | Arshdeep [Pdf]() | 
| Scalable | Hierarchical graph representation learning with differentiable pooling  | [PDF](https://arxiv.org/abs/1806.08804)   | Eamon [Pdf]() | Ryan [Pdf]() | 
---
layout: post
title: GNN   
desc: 2019-W11
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---


| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Program |   Neural network-based graph embedding for cross-platform binary code similarity detection  | [Pdf](https://openreview.net/forum?id=BJOFETxR-) + [Pdf](https://arxiv.org/abs/1708.06525)| Faizan [Pdf]() | GaoJi [Pdf]() | 
| Basics | MILE: A Multi-Level Framework for Scalable Graph Embedding + Learning to represent programs with graphs  | [Pdf](https://arxiv.org/abs/1802.09612) | Ryan [Pdf]() | Derrick [Pdf]() |
| Geometric | Dynamic graph cnn for learning on point clouds, 2018, cite67  | [Pdf](https://arxiv.org/abs/1801.07829) | Fuwen [Pdf]() | Arshdeep [Pdf]() | 

---
layout: post
title: GNN   
desc: 2019-W12
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---


| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Basics |  Link Prediction Based on Graph Neural Networks + Supervised Community Detection with Line Graph Neural Networks | [Pdf](https://arxiv.org/abs/1802.09691) + [More](https://paperswithcode.com/task/graph-embedding) | Jack [Pdf]() |  | 
|  Understand |  Interpretable Graph Convolutional Neural Networks for Inference on Noisy Knowledge Graphs +  GNN Explainer: A Tool for Post-hoc Explanation of Graph Neural Networks | [Pdf](https://arxiv.org/abs/1812.00279) + [PDF](https://arxiv.org/abs/1903.03894) | GaoJi [Pdf]() |  | 
| Generate |   Inference in probabilistic graphical models by Graph Neural Networks + Encoding robust representation for graph generation | [Pdf](https://arxiv.org/abs/1809.10851)+ [PDF](https://arxiv.org/abs/1803.07710) |  Arshdeep [Pdf]() |  | 
|  QA | Learning to Reason Science Exam Questions with Contextual Knowledge Graph Embeddings / Knowledge Graph Embedding via Dynamic Mapping Matrix    | [PDF](http://www.aclweb.org/anthology/P15-1067) + [Pdf](https://arxiv.org/abs/1805.12393) | Bill [Pdf]() | GaoJi [Pdf]() | 



---
layout: post
title: GNN   
desc: 2019-W13
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---


| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
|  Bio |  Molecular geometry prediction using a deep generative graph neural network  | [Pdf](https://arxiv.org/abs/1904.00314) | Eli [Pdf]() | Jack [Pdf]() |
|  Scalable |     Towards Efficient Large-Scale Graph Neural Network Computing     | [Pdf](https://arxiv.org/abs/1810.08403) | Derrick [Pdf]() |  | 
| Generate |  Convolutional Imputation of Matrix Networks  / Graph Convolutional Matrix Completion | [Pdf](http://proceedings.mlr.press/v80/sun18d/sun18d.pdf) | Tkach [Pdf]() | Arshdeep [Pdf]() | 
---
layout: post
title: GNN   
desc: 2019-W14
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---

| Index | Papers Info | Paper URL| Our Summary Slide |Our Scribe Note |
| -----: | -------------------------------: | :----- | :----- | :----- | 
| Edge |  Espresso: Efficient Forward Propagation for Binary Deep Neural Networks    | [Pdf](https://arxiv.org/abs/1705.07175) | Eamon [Pdf]() | Derrick [Pdf]() | 
| Robust |  Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution | [PDF](https://arxiv.org/abs/1904.05049) | Weilin [Pdf]() |  | 
| Basics | LanczosNet: Multi-Scale Deep Graph Convolutional Networks  | [Pdf](https://openreview.net/forum?id=BkedznAqKQ) | Ryan [Pdf]() | Arshdeep [Pdf]() |
| Robust |   Attacking Binarized Neural Networks    | [Pdf](https://openreview.net/forum?id=HkTEFfZRb) | Faizan [Pdf]() | GaoJi [Pdf]() | 
| Geometric |  Geometric matrix completion with recurrent multi-graph neural networks     | [Pdf](https://arxiv.org/abs/1704.06803) | Fuwen [Pdf]() | Arshdeep [Pdf]() | 

---
layout: post
title: GNN Project presenations in 109  
desc: 2019-W15
categories: 2019sCourse
tags:
- 7Graphs
---


| Index | Time | Presenter | Slides |
| -----: | -----: | ---------: | -------------------------------: |  
| 1| 9:20 am | weilin |    |  
| 2| 10:00 am | Fuwen + Arsh |    |  
| 4| 10:30am | Bill |    |  
| 5| 11am | GaoJi |    |  
| 6| 11:30am | Derrick |    |  
| 7| 12pm |  Ryan |   |  
| 8| 12:30pm | Eli+ Jack |    |  
| 9| 1:10pm~2pm | LUNCH |    |  
| 10| 2 pm  | Tkach |    |  
| 11| 2:30 pm | Eamon |    |  
| 12| 3pm  | Brandon |    |  
| 13| 3:30pm | ZhaoYang |    |  
| 14| 4pm | Faizan Video |    |  
| 15| 4:30pm | Jennifer Video |    |  
---
layout: post
title: GNN Project Day Reads
desc: 2019-W buffer
categories: 2019sCourse
tags:
- 7Graphs
gnnType: basics
---



| Presenter | Papers | Paper URL| Our Presentation | 
| -----: | -------------------------------: | :----- | :----- | 
|  &#9745;  Bill |  MatchMiner: An open source computational platform for real-time matching of cancer patients to precision medicine clinical trials using genomic and clinical criteria. (2017)  |     
| &#9745; GaoJi | DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning | | |
| Eamon | Attention is not Explanation, 2019   | [PDF](https://arxiv.org/abs/1902.10186)   |  | 
| Jack | Convolutional Set Matching for Graph Similarity | | |  
| Jack | Querying complex networks in vector space | 
| &#9745; Eli | Visualizing convolutional neural network protein-ligand scoring | | | 
| &#9745; Eli | Deep generative models of genetic variation capture mutation effects |     |  |  
| &#9745; Eli |  Attentive cross-modal paratope prediction   [Pdf](https://openreview.net/forum?id=ByUU2t1PG) |  |  |  
| Derrick | Dynamic Scheduling For Dynamic Control Flow in Deep Learning Systems   | [PDF](http://www.cs.cmu.edu/~jinlianw/papers/dynamic_scheduling_nips18_sysml.pdf) |  |  
| &#9745; Derrick | Cavs: An Efficient Runtime System for Dynamic Neural Networks    [Pdf](https://www.usenix.org/system/files/conference/atc18/atc18-xu-shizhen.pdf) | |
| Ryan |  MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications | [PDF]()   |  | 
| &#9745;  Ryan | XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks | [PDF]()   |  | 
| &#9745; Arsh |  Geometric Deep Learning on Graphs and Manifolds - #NIPS2017  [URL](https://www.youtube.com/watch?v=LvmjbXZyoP0)  |   |  | 
| &#9745; Arsh | Similarity Learning with Higher-Order Proximity for Brain Network Analysis | | | 
| Arsh | Multi-Task Graph Autoencoders  |  | | 
| &#9745; Arsh| Modeling Relational Data with Graph Convolutional Networks | [PDF](https://arxiv.org/abs/1703.06103) | |  
| Faizan | Adversarial Text Generation via Feature-Mover's Distance | | |
| &#9745; Faizan | Content preserving text generation with attribute controls   |  |  |  
| &#9745; Faizan | Multiple-Attribute Text Rewriting, ICLR, 2019, | | |
| &#9745; Tkach |  NetGAN: Generating Graphs via Random Walks ICML18 | | |
| &#9745; Tkach |  Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation NeurIPS18 |  | |
| &#9745; Tkach |  Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement | | |
| &#9745; Fuwen | Pixel to Graph with Associative Embedding | [PDF]()     |  
| &#9745; Fuwen | 3d steerable cnns: Learning rotationally equivariant features in volumetric data  |  
