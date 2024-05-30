<div align="center">
<h1>
CheXpert Plus: Hundreds of Thousands of Aligned Radiology Texts, Images, and Patients
</h1>
</div>

## Table of Content

- [RadGraph-XL](#RadGraph-XL)
- [CheXbert](#chexbert)
- [Model Zoo](#model-zoo)

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
df = pd.read_csv('df_chexpert_plus_240401.csv')
df = df[df['section_findings'].apply(lambda x: isinstance(x, str) and len(x.split()) >= 2)]

# Load radgraph-XL annotations
with open("radgraph-XL-annotations/section_findings.json") as f:
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
df = pd.read_csv('df_chexpert_plus_240401.csv')

# Load radgraph-XL annotations
with open("radgraph-XL-annotations/section_findings.json") as f:
    annotations = json.load(f)

annotations_dict = {a["0"]["text"]: a for a in annotations}

# Get a random findings:
findings = df.iloc[20]["section_findings"]
preprocessed_findings = utils.radgraph_xl_preprocess_report(findings)

# Retrieve annotation
print(annotations_dict[preprocessed_findings])
```

## CheXbert

The json files contains the mapping between CheXpert images and diseases extracted from the radiology report section.

```python
import json

json_diseases = [json.loads(s) for s in open("chexbert_labels/findings_fixed.json").readlines()]
print(json.dumps(json_diseases[0], indent=4))
> {
    "path_to_image": "train/patient42142/study5/view1_frontal.jpg",
    "Enlarged Cardiomediastinum": null,
    "Cardiomegaly": null,
    "Lung Opacity": null,
    "Lung Lesion": null,
    "Edema": null,
    "Consolidation": null,
    "Pneumonia": null,
    "Atelectasis": null,
    "Pneumothorax": null,
    "Pleural Effusion": null,
    "Pleural Other": null,
    "Fracture": null,
    "Support Devices": null,
    "No Finding": 1.0
}
```

You can merge these diseases annotations with the main CSV:

```python
import json
import pandas as pd

# Merge both DataFrames on the 'path_to_image' column
jsonl_df = pd.read_json('chexbert_labels/findings_fixed.json', lines=True)
csv_df = pd.read_csv('df_chexpert_plus_240401.csv')
merged_df = pd.merge(jsonl_df, csv_df, on='path_to_image')

# Filter DataFrame to include only rows where 'section_findings' is not null
filtered_df = merged_df[merged_df['section_findings'].notna()]
```

You can further create the mapping findings -> diseases:

```python
disease_columns = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]

# Create a dictionary where the key is 'section_findings' and the value is mapping diseases
findings_to_diseases = {
    row['section_findings']: {disease: row[disease] for disease in disease_columns}
    for index, row in filtered_df.iterrows()
}

print(json.dumps(findings_to_diseases, indent=4))
> {
    "Unchanged right internal jugular venous catheter. Stable ....":
        {
            "Enlarged Cardiomediastinum": -1.0,
            "Cardiomegaly": NaN,
            "Lung Opacity": 1.0,
            "Lung Lesion": NaN,
            "Edema": 1.0,
            "Consolidation": NaN,
            "Pneumonia": -1.0,
            "Atelectasis": -1.0,
            "Pneumothorax": NaN,
            "Pleural Effusion": NaN,
            "Pleural Other": NaN,
            "Fracture": NaN,
            "Support Devices": 1.0,
            "No Finding": NaN
        },
    ...
}
```

## Model Zoo

| Type   | Datasets                                                  | Model                                                                                                                 | Link                                                                            | Tutorial                                                                                             |
|--------|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| RRG    | MIMIC-cxr & Chexpert Plus                                 | [Swinv2](https://huggingface.co/docs/transformers/en/model_doc/swinv2#transformers.Swinv2Model)/bert-decoder-2-layers | [🤗](tutorials/RRG/INDEX.md)                                                    | [Doc](tutorials/RRG/INDEX.md)                                                                        
| VQGAN  | MIMIC-CXR & CheXpert Plus & PadChest & BIMCV & Candid-PTX | XrayVQGAN                                                                                                             | [🤗](https://huggingface.co/StanfordAIMI/XrayVQGAN)                             | [Doc](https://github.com/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb) | 
| DINOv2 | MIMIC-CXR & CheXpert Plus & PadChest & BIMCV & Candid-PTX | XrayDINOv2                                                                                                            | [🤗](https://huggingface.co/StanfordAIMI/dinov2-base-xray-224)                  | [Doc](https://huggingface.co/docs/transformers/model_doc/dinov2)                                     |
| CLIP   | MIMIC-CXR & CheXpert Plus & PadChest & BIMCV & Candid-PTX | XrayCLIP                                                                                                              | [🤗](https://huggingface.co/StanfordAIMI/XrayCLIP__vit-b-16__laion2b-s34b-b88k) | [Doc](https://huggingface.co/docs/transformers/model_doc/clip)                                       |
| LLaMA  | -                                                         | RadLLaMA                                                                                                              | [🤗](https://huggingface.co/StanfordAIMI/RadLLaMA-7b)                           | [Doc](tutorials/radllama/README.md)                                                                  | 

## Reference

```
@article{chexpert-plus-2024,
  title={CheXpert Plus: Hundreds of Thousands of Aligned Radiology Texts, Images and Patients},
  author={Pierre Chambon, Jean-Benoit Delbrouck, Thomas Sounack, Shih-Cheng Huang, Zhihong Chen, Maya Varma, Steven QH Truong, Chu The Chuong, Curtis P. Langlotz},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  url={https://arxiv.org/abs/xxxx.xxxxx},
  year={2024}
}
```
