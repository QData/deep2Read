---
layout: post
title: GNN for BioMed Applications   
desc: 2019-W3
term: 2019sCourse
categories:
- 2Graphs
- 9DiscreteApp
tags: [ attention, relational, visualizing, geometric, DNA, protein, molecule]
---


| Presenter | Papers | Paper URL| Our Slides |
| -----: | -------------------------------------: | :----- | :----- |
|  Bio |  KDEEP: Protein–Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks, 2018 [^2] |  [Pdf](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00650) | Eli [Pdf]({{site.baseurl}}/talks2019/19sCourse/20190315-Eli-Kdeep.pdf)  | 
|  Bio |  Molecular geometry prediction using a deep generative graph neural network  | [Pdf](https://arxiv.org/abs/1904.00314) | Eli [Pdf]({{site.baseurl}}/talks2019/19sCourse/20190419-Eli-MolecularGeometryVAE.pdf)  |
| Bio | Visualizing convolutional neural network protein-ligand scoring |  PDF() | Eli [PDF]({{site.baseurl}}/talks2019/Extra19s/EliVisualizeCNNProtein.pdf) | 
| Bio | Deep generative models of genetic variation capture mutation effects | PDF() | Eli [PDF]({{site.baseurl}}/talks2019/Extra19s/EliGenerativeVariants.pdf)  |  
| Bio |  Attentive cross-modal paratope prediction |  [Pdf](https://openreview.net/forum?id=ByUU2t1PG) |  Eli [PDF]({{site.baseurl}}/talks2019/Extra19s/ELiAttentiveAB.pdf)  |  


<!--excerpt.start-->
[^2]: <sub><sup> Jack Note:  Accurately predicting protein−ligand binding affinities is an important problem in computational chemistry since it can substantially accelerate drug discovery for virtual screening and lead optimization. This paper proposes using 3D-CNNs for predicting binding affinities across several diverse datasets. The main idea is they represent the protein and ligand together using their 3D voxel representation. This complex representation of the protein-ligand together is fed through a 3D-CNN to produce a scalar affinity value. It is trained using MSE on the ground truth affinity values. The authors use 4 datasets: PDB containing 58 and 290 complexes, CSAR NRC-HiQ containing 167 and 176 complexes, CSAR2012 containing 57, and CSAR2014 containing 47.   The authors use a 3D voxel representation of both proteins and ligand using a van der Waals radius for each atom type, which in turns gets assigned to a particular property channel (hydrophobic, hydrogen-bond donor or acceptor, aromatic, positive or negative ionizable, metallic and total excluded volume), according to certain rules. The contribution of each atom to each grid point depends on their Euclidean distance. They duplicate the number of properties to account for both protein and ligand, by using the same ones in each, resulting in up to a total of 16 different channels. Their 3D-CNN performed well compared to previous methods and resulted in speed increases due to the parallelization of the GPU.  However, it seems the biggest concern is the representation of the protein-ligand complex considering a specific docking tool is needed to specify how the protein and ligan and linked in the voxel space. I feel this severely prohibits the model when training considering no perturbations of the docking are used. On a similar note, it's very hard to define ``negative'' samples in this task, and I'm curious to see how their model would predict a completely incorrect, or negative protein-ligand complex. <sup><sub>
