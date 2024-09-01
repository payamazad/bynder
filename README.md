# Business Justifications
This product is an "asset enrichment" tool.  The final product supposed to provide a tool that whenever a user uploads an
image our system would enrich it with extracting (predicting) metadata. 
## Proposed usage
Proposed usage for this product is a scalable microservice, managed by Kubernetes, that leverages GPU-enhanced VMs to run 
Docker images of  our prediction service. This service will accept images in various formats (e.g., JPG, PNG) and return metadata as JSON.
## Output format
This output format is based on the provided (test) dataset and my analysis over it. These are fields that we can predict:
- gender
- baseColour
- season
- usage
- articleType (we can extract masterCategory and subCategory using a simple mapping out of articleType)
- productDisplayName: This will be a description of image from a on-purpose trained model or structured text from previous extracted metadata (I could not finish this part)

# Implementation
## Prediction model
This project involves a multi-label image classification task, aiming to predict five labels per image. Additionally, 
we're tackling an image captioning problem by connecting a pre-trained model (like one trained on ImageNet) to a text 
generator. To tailor the model to our specific needs, we'll fine-tune it on our dataset.

So, the base for these models will be an image classification model, pretrained on a bigger image dataset; and I will
try finetune this model for Bynder dataset.

### Model selection
The field of image processing is witnessing a significant shift towards models based on Attention and Transformer 
architectures. These approaches consistently outperform previous leaders built on Convolutional Neural Networks (CNNs). 
Prominent examples include [CoCa](https://arxiv.org/pdf/2205.01917v2) and [CoAtNet](https://arxiv.org/pdf/2106.04803v2)
from Google, and [SwinV2](https://arxiv.org/pdf/2111.09883v2) from Microsoft. However, these cutting-edge models 
come with a significant drawback: their immense size often exceeding 2 billion parameters. The sheer size of these SOTA 
models presents several challenges for real-world applications. Training them requires powerful hardware setups like 
clusters of GPUs and TPUs, making them inaccessible to most users. Additionally, performing predictions with such models 
incurs high computational costs.

The provided table effectively compares models based on their accuracy 
(e.g., ImageNet performance) and parameter count. This comparison serves as a crucial tool for selecting an appropriate 
model, striking a balance between performance and computational feasibility.

| Model           | Accuracy | #Parameters |
| :--------       |---------:|------------:|
| [Coca](https://arxiv.org/pdf/2205.01917v2)            |       91 |       2100M |
| [CoAtNet](https://arxiv.org/pdf/2106.04803v2)         |    90.88 |       2440M |
| [SwinV2](https://arxiv.org/pdf/2111.09883v2)          |    90.17 |       3000M |
| [Distilled Vit](https://arxiv.org/pdf/2302.05442v1)   |     89.6 |        307M |
| [ViT-B/16](https://arxiv.org/pdf/2302.05442v1)        |     88.6 |         86M |
| [CAFormer](https://arxiv.org/pdf/2210.13452v3)        |     86.9 |         39M |
| [TinyViT](https://arxiv.org/pdf/2207.10666v1)         |     86.5 |         21M |

As you can see there is no huge difference in quality of 2B+ parameter model and 100M parameter models. 
In production based on the cost per quality optimisation we can go with any of the models under 300M parameters (e.g. Distilled ViT)
but for the sake of this task I have chosen TinyVit-22m-distilled that is a distilled version of Vision Transformer 
with 22m parameters developed by [microsoft](https://github.com/microsoft/Cream/tree/main/TinyViT). 

### Distillation
[Distillation](https://arxiv.org/abs/1503.02531) is a very strong technique that is used to create smaller models with 
limited resources. In this technique a bigger model (teacher) is trained on huge dataset is used to teach its knowledge
to a smaller model (student) by utilizing teacher model's output as soft-labels for student model and student model to 
try to learn whatever teacher has learnt. By, this technique we can transform big part of information from learned by teacher model
to student model and make a smaller models that are specialized on a specific dataset but has a general (one-shot learning) knowledge.
This technique has used to improve and train TinyViT. A swine2, model that is trained on a huge dataset (internal to microsoft) has been 
used as a teacher for tinyViT finetuning on ImageNet (that is itself a huge dataset). TinyVit-22m-distilled 


### Base model discussion
### Captioning
## Training
### Data split
### 
## Deployment
Docker over GPU...
### Scaling
Kubernetes

# Usage
## environment setup guide
## code structure
# Future Improvements

# Links
## best classification models
[CoCa](https://paperswithcode.com/paper/coca-contrastive-captioners-are-image-text)
[CoCa paper](https://arxiv.org/pdf/2205.01917v2)
[Meta Pseudo Labels](https://paperswithcode.com/paper/meta-pseudo-labels)
[Meta Pseudo Labels paper](https://arxiv.org/pdf/2003.10580v4)