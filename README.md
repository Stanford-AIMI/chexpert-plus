<div align="center">
<h1>
CheXpert Plus: Hundreds of Thousands of Aligned Radiology Texts, Images, and Patients
</h1>
</div>

## Table of Content
- [RadGraph-XL](#RadGraph-XL)
- [CheXbert](#chexbert)
- [Data](#data)
- [Model](#model)

## RadGraph-XL

First install radgraph-XL through the [radgraph](https://pypi.org/project/radgraph/) package:
```bash
pip install radgraph
```

There are two ways to access the radgraph-XL annotation from the CSV. First, using row index:
```python
import json
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('data/chexpert/files/df_chexpert_plus_240401.csv')
df = df[df['section_findings'].apply(lambda x: isinstance(x, str) and len(x.split()) >= 2)]

# Load radgraph-XL annotations
with open("/home/jb/Downloads/radgraph-XL-annotations/section_findings.json") as f:
    annotations = json.load(f)

# get fifth example
index = 5
findings = df.iloc[index]["section_findings"]
annotation = annotations[index]
print(findings)
print(annotation)
```
Or directly using the section content:
```python
import json
import pandas as pd
from radgraph import utils

# Load the CSV file into a DataFrame
df = pd.read_csv('data/chexpert/files/df_chexpert_plus_240401.csv')

# Load radgraph-XL annotations
with open("/home/jb/Downloads/radgraph-XL-annotations/section_findings.json") as f:
    annotations = json.load(f)

annotations_dict = {a["0"]["text"]: a for a in annotations}

# Get a random findings:
findings = df.iloc[20]["section_findings"]
preprocessed_findings = utils.radgraph_xl_preprocess_report(findings)

# Retrieve annotation
print(annotations_dict[preprocessed_findings])
```


## Data

## Model

| Type   | Datasets        | Model      | Link                                                                            | Tutorial                                                                                             |
|--------|-----------------|------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| RRS    |                 |            |                                                                                 |                                                                                                      |
| RRG    |                 |            |                                                                                 |                                                                                                      |
| VQGAN  | Xray Collection | XrayVQGAN  | [🤗](https://huggingface.co/StanfordAIMI/XrayVQGAN)                             | [Doc](https://github.com/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb) | 
| DINOv2 | Xray Collection | XrayDINOv2 | [🤗](https://huggingface.co/StanfordAIMI/dinov2-base-xray-518)                  | [Doc](https://huggingface.co/docs/transformers/model_doc/dinov2)                                     |
| CLIP   | Xray Collection | XrayCLIP   | [🤗](https://huggingface.co/StanfordAIMI/XrayCLIP__vit-b-16__laion2b-s34b-b88k) | [Doc](https://huggingface.co/docs/transformers/model_doc/clip)                                       |
| LLaMA  | Clinical Corpus | RadLLaMA   | [🤗](https://huggingface.co/StanfordAIMI/RadLLaMA-7b)                           | [Doc](tutorials/radllama/README.md)                                                                  | 
