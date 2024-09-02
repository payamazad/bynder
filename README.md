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
| [TinyViT](https://arxiv.org/pdf/2207.10666v1)         |     80.5 |          5M |

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
used as a teacher for tinyViT finetuning on ImageNet (that is itself a huge dataset). TinyVit-5m-distilled is using this
strong technique to make efficient mdoels with just 5m parameters.

## Training
To address the multi-classification problem, I initially explored a multi-output model with a combined loss function. 
However, this approach led to suboptimal performance in predicting "articleType" and "baseColor." Despite potential 
benefits in terms of model storage and inference efficiency, I opted for a more modular approach with individual models 
for each classification task. This allowed for greater control over model architecture and optimization, ensuring adequate 
performance for all categories. While data augmentation and model partitioning were considered, optimizing the loss 
function to prioritize "articleType" and "baseColor" was deemed the most promising solution. Due to time constraints, 
I ultimately proceeded with separate models for each classification task.

To fine-tune the model, I initially froze all layers except the final fully-connected layer. However, training the 
entire model for a single output (gender) resulted in a minimal improvement in loss (3.3%) at the cost of significantly 
slower training time (4.2 times slower). Based on this, I decided to continue training only the last layer, which offers 
a balance between performance gains and computational efficiency. 

I have trained each model for 10 epochs and saving the best model and finally loading the best model.

### Data split
I have used %70 of the data for training and %10 for validation. I have used remaining %20 for testing and reporting the results.


# Usage
## environment setup guide
Current codebase is written using python 3.10 and latest versions of packages. For installing packages you need to run:
```
pip install -r requirements.txt
```

For running trainings you need to call 
```
python run.py
``` 
This will create datasets and start training models.

To create docker image and serve it you need to go to directory bynder/src/torch_serve and run:
```
docker build -t multi_model_torchserve .
docker run -p 8080:8080 -p 8081:8081 multi_model_torchserve
``` 

Then you can call this server using 
```
curl -X POST http://localhost:8080/predictions/metadata_creator -T image_path.jpg
```


## code structure
There are several folders in this package:
- logs: include training and testing metric logs for each model
- models: include models saved for each label.
- src: including python codes
  - data.py is the DataLoader classes
  - model.py includes image classification models definitions with train, test, save and load functions
  - Cream: is the package by Microsoft that includes TinyViT code 
  - torch_serve: is the folder including torch serve setup codes
    - model_store: is the place that we store .mar models to serve
    - create_mar: is the script for creating mar files from traced models
    - Dockerfile: is the docker file to create a docker including torch serve

# Future Improvements
List of improvements that we can deploy to get better results:
1. Using a stronger model (I would go Distilled-ViT from Google)
2. Augmenting images, to get a more diverse dataset
3. Predict baseColour using histogram of colors as feature and a sample Logistic Regression
4. Distilling whole model using another stronger model (I would select CoCa as teacher)
5. Analyse data deeper and find class-imbalance cases and tackle them (by updating loss, merging classes, SMOTE or other techniques)
6. Normalization and standardization of images
7. Train for longer number of epochs (combined with augmentation)
8. Adding early stopping
9. Create multi-output model (for improving inference speed and cost optimization)
10. Better training script with early-stopping and optimizer scheduler
11. Handling missing labels better