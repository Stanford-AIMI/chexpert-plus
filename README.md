<div align="center">
<h1>
CheXpert Plus: Hundreds Thousands of Aligned Radiology Texts, Images, and Patients
</h1>
</div>

## Table of Content

- [Data](#data)
- [Model](#model)

## Data

## Model

| Type   | Datasets        | Model      | Link                                                                            | Tutorial                                                                                             |
|--------|-----------------|------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| RRS    |                 |            |                                                                                 |                                                                                                      |
|        |                 |            |                                                                                 |                                                                                                      |
| RRG    |                 |            |                                                                                 |                                                                                                      |
| VQGAN  | Xray Collection | XrayVQGAN  | [🤗](https://huggingface.co/StanfordAIMI/XrayVQGAN)                             | [Doc](https://github.com/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb) | 
| DINOv2 | Xray Collection | XrayDINOv2 | [🤗](https://huggingface.co/StanfordAIMI/dinov2-base-xray-518)                  | [Doc](https://huggingface.co/docs/transformers/model_doc/dinov2)                                     |
| CLIP   | Xray Collection | XrayCLIP   | [🤗](https://huggingface.co/StanfordAIMI/XrayCLIP__vit-b-16__laion2b-s34b-b88k) | [Doc](https://huggingface.co/docs/transformers/model_doc/clip)                                       |
| LLaMA  | Clinical Corpus | RadLLaMA   | [🤗](https://huggingface.co/StanfordAIMI/RadLLaMA-7b)                           | [Doc](https://huggingface.co/docs/transformers/model_doc/llama)                                      | 