---
desc: 2022-W5
term: 2022-selfRead
title: Stable diffusion + DreamBooth + LoRA  
categories:
- FMBasic
- FMMulti
tags: [ Diffusion, Image synthesis, Efficiency ]
---
 

## Stable diffusion 
+ [ URL](https://arxiv.org/abs/2112.10752) 
+ "High-Resolution Image Synthesis with Latent Diffusion Models"


## DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation 
+ [ URL](https://arxiv.org/abs/2208.12242) 
+  "personalization" of text-to-image diffusion models. Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model such that it learns to bind a unique identifier with that specific subject. ." 


## LoRA: Low-Rank Adaptation of Large Language Models 
+ [  URL](https://arxiv.org/abs/2106.09685) 
+ "propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times." |


## An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion
+ https://arxiv.org/abs/2208.01618
+ Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, Daniel Cohen-Or
+ Text-to-image models offer unprecedented freedom to guide creation through natural language. Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes. In other words, we ask: how can we use language-guided models to turn our cat into a painting, or imagine a new product based on our favorite toy? Here we present a simple approach that allows such creative freedom. Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model. These "words" can be composed into natural language sentences, guiding personalized creation in an intuitive way. Notably, we find evidence that a single word embedding is sufficient for capturing unique and varied concepts. We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks.