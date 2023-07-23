# WSCL
<small>_**W**eak **S**upervise and **C**ontrastive **L**earning method on Cervical Malignant Classification with Inaccurate Labels._</small>

Work flow of our work is presented below.

![](./pics/work_flow.png)

Our method consist two stages: 

a. **Feature Learning Stage**. We use a simple _ResNet50_ to extract features. In this stage we train the extractor in the file "train_simclr.py". 

b. **Classifier Finetuning stage**. Firstly we generate pseudo labels for _Malignant_ images based on ensemble predictions and sort their prototypes by the ensemble confidences (in "generate_pseudo.py"). Then we choose topk (128) prototypes to calibrate the generated logits for  _Malignant_ type, acquring the final prediction results (in "finetune.py").


Except for the two training files and one pseudo labeling file mentioned above, we aslo provide:

----data

------------augmentation.py <small>(data augmentations for two stages)</small>

------------gaussian_blur.py <small>(gaussian blur augmentation for the first stage)</small>

------------loader.py <small>(dataloader for two stages)</small>

----model

------------resnet_simclr.py <small>(network architecture for feature extraction and classification based on ResNet)</small>

----utils

------------ema.py <small>(model exponential moving average relaization)</small>

------------loss.py <small>(re-weight cross-entropy loss and self-supervised contrastive loss)</small>

------------train.py <small>(tools for training and validating, including adjusting learning rate, record losses, reading prototypes and calculate accuracy)</small>

----evaluation.py <small>(for validating)</small>

----vis_cam.py <small>(visualize class activation maps)</small>
      