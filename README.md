# Business Justifications
This product is a "asset enrichment" tool.  The final product supposed to provide a tool that whenever a user uploads an image our system would enrich it with extracted (predicted) metadata. 
## Proposed usage
This will be a scalable micro-service (e.g. managed by Kubernetes) that would load our prediction service as Docker images (from Dockerfile) over (preferably) GPU enhanced VMs. This micro-service will recieve an image in jpg, png or other formats and returns a metadata as json. 
## Output format
This output format is based on the provided (test) dataset and my analysis over it. These are fields that we can predict:
- gender
- baseColour
- season
- usage
- articleType (we can extract masterCategory and subCategory using a simple mapping out of articleType)
- productDisplayName: This will be a description of image from a on-purpose trained model or structured text from previous extracted metadata

## Input perfromance improvement suggestion
For improving performance of all service, we can just provide the commonly reachable address of image to service instead of uploading image itself. So, after uploading the image to an storage provide address of it to the service and it will read the image from storage and do the predictions.  

# Implementation
## Prediction model
We are dealing with specific image classification problem that suppose to predict [multi-labels](## Output format) (5 output) and also a captioning problem that can be solved with connecting a classifiers one of the latest layers before softmax to a text creator. We need to take a general purpose model (trained on huge datasets like Imagenet) and fine tune to adapt to our problem.

### Model selection


Latest SOTA models are mainly based on Attention and Tranformer architecture. These architecures easily surpass previous SOTAs that were based on CNNs. There are severa


Problem types: 
- Image classification
- multi label
- meta learning (Few Shot Learning)
- image captioning
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