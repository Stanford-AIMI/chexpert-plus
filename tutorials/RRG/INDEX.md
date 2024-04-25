## Model Zoo

| Datasets        | Model      | Link |                                                                                   
|-----------------|------------|---------------------------------------------------------------------------------|
 |   mimic-cxr-findings              |  [Swinv2](https://huggingface.co/docs/transformers/en/model_doc/swinv2#transformers.Swinv2Model)/bert-decoder-2-layers          |      [🤗](https://huggingface.co/IAMJB/mimic-cxr-findings-baseline)        |
 |   mimic-cxr-impression              |  [Swinv2](https://huggingface.co/docs/transformers/en/model_doc/swinv2#transformers.Swinv2Model)/bert-decoder-2-layers          |      [🤗](https://huggingface.co/IAMJB/mimic-cxr-impression-baseline)        |
  |   chexpert-findings              |  [Swinv2](https://huggingface.co/docs/transformers/en/model_doc/swinv2#transformers.Swinv2Model)/bert-decoder-2-layers          |      [🤗](https://huggingface.co/IAMJB/chexpert-findings-baseline)        |
   |   chexpert-impression              |  [Swinv2](https://huggingface.co/docs/transformers/en/model_doc/swinv2#transformers.Swinv2Model)/bert-decoder-2-layers          |      [🤗](https://huggingface.co/IAMJB/chexpert-impression-baseline)        |
 |   mimic-cxr-chexpert-findings              |  [Swinv2](https://huggingface.co/docs/transformers/en/model_doc/swinv2#transformers.Swinv2Model)/bert-decoder-2-layers          |      [🤗](https://huggingface.co/IAMJB/chexpert-mimic-cxr-findings-baseline)        |
 |   mimic-cxr-chexpert-impression              |  [Swinv2](https://huggingface.co/docs/transformers/en/model_doc/swinv2#transformers.Swinv2Model)/bert-decoder-2-layers          |      [🤗](https://huggingface.co/IAMJB/chexpert-mimic-cxr-impression-baseline)        |

 ## Usage

 ```python
import torch
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor, VisionEncoderDecoderModel, GenerationConfig
import requests

mode = "impression"
# Model
model = VisionEncoderDecoderModel.from_pretrained(f"IAMJB/chexpert-mimic-cxr-{mode}-baseline").eval()
tokenizer = BertTokenizer.from_pretrained(f"IAMJB/chexpert-mimic-cxr-{mode}-baseline")
image_processor = ViTImageProcessor.from_pretrained(f"IAMJB/chexpert-mimic-cxr-{mode}-baseline")
#
# Dataset
generation_args = {
    "bos_token_id": model.config.bos_token_id,
    "eos_token_id": model.config.eos_token_id,
    "pad_token_id": model.config.pad_token_id,
    "num_return_sequences": 1,
    "max_length": 128,
    "use_cache": True,
    "beam_width": 2,
}
#
# Inference
refs = []
hyps = []
with torch.no_grad():
    url = "https://huggingface.co/IAMJB/interpret-cxr-impression-baseline/resolve/main/effusions-bibasal.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    # Generate predictions
    generated_ids = model.generate(
        pixel_values,
        generation_config=GenerationConfig(
            **{**generation_args, "decoder_start_token_id": tokenizer.cls_token_id})
    )
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_texts)
```

 
## Evaluations

 
| Metric                          | MIMIC-CXR-VAL | CHEXPERT-VAL | MIMIC-TEST | CHEXPERT-TEST |
|---------------------------------|---------------|--------------|------------|---------------|
| **mimic-cxr-findings**          |               |              |            |               |
| ROUGEL                          | 28.33         | 19.08        | 25.36      | 21.49         |
| bertscore                       | 56.22         | 46.26        | 53.51      | 48.76         |
| radgraph_simple                 | 30.54         | 22.80        | 27.43      | 23.64         |
| radgraph_partial                | 27.30         | 19.82        | 24.43      | 21.42         |
| radgraph_complete               | 21.21         | 13.58        | 18.10      | 15.26         |
| BLEU                            | 6.37          | 3.39         | 5.73       | 3.33          |
| chexbert-5_micro avg_f1-score   | 53.06         | 42.25        | 55.30      | 45.08         |
| chexbert-all_micro avg_f1-score | 49.26         | 40.96        | 49.72      | 45.59         |
| chexbert-5_macro avg_f1-score   | 45.45         | 38.31        | 47.04      | 39.42         |
| chexbert-all_macro avg_f1-score | 33.04         | 26.81        | 32.16      | 30.39         |
| **mimic-cxr-impression**        |               |              |            |               |
| ROUGEL                          | 30.31         | 11.29        | 23.37      | 11.42         |
| bertscore                       | 51.30         | 37.45        | 46.44      | 37.60         |
| radgraph_simple                 | 31.66         | 11.08        | 26.70      | 12.16         |
| radgraph_partial                | 29.16         | 9.63         | 23.51      | 10.77         |
| radgraph_complete               | 22.82         | 5.61         | 18.33      | 7.31          |
| BLEU                            | 3.44          | 0.62         | 2.78       | 0.81          |
| chexbert-5_micro avg_f1-score   | 52.03         | 41.85        | 53.01      | 41.13         |
| chexbert-all_micro avg_f1-score | 53.34         | 39.68        | 48.37      | 39.58         |
| chexbert-5_macro avg_f1-score   | 44.90         | 37.03        | 45.88      | 34.99         |
| chexbert-all_macro avg_f1-score | 34.56         | 25.41        | 33.15      | 28.36         |
| **chexpert-findings**           |               |              |            |               |
| ROUGEL                          | 22.35         | 22.82        | 19.57      | 23.78         |
| bertscore                       | 50.90         | 50.66        | 48.04      | 51.35         |
| radgraph_simple                 | 24.73         | 28.54        | 22.81      | 29.23         |
| radgraph_partial                | 23.11         | 25.01        | 20.69      | 26.06         |
| radgraph_complete               | 15.04         | 18.33        | 13.85      | 19.35         |
| BLEU                            | 2.33          | 5.69         | 2.22       | 5.04          |
| chexbert-5_micro avg_f1-score   | 39.98         | 43.17        | 45.41      | 49.01         |
| chexbert-all_micro avg_f1-score | 42.78         | 50.78        | 43.98      | 50.69         |
| chexbert-5_macro avg_f1-score   | 34.47         | 40.79        | 39.67      | 43.50         |
| chexbert-all_macro avg_f1-score | 28.00         | 33.07        | 28.99      | 29.94         |
| **chexpert-impression**         |               |              |            |               |
| ROUGEL                          | 14.86         | 27.94        | 14.05      | 27.26         |
| bertscore                       | 38.80         | 53.02        | 38.00      | 53.32         |
| radgraph_simple                 | 17.47         | 28.47        | 16.94      | 27.07         |
| radgraph_partial                | 15.46         | 25.73        | 14.94      | 24.57         |
| radgraph_complete               | 9.76          | 19.43        | 10.06      | 18.65         |
| BLEU                            | 1.97          | 6.85         | 1.80       | 6.55          |
| chexbert-5_micro avg_f1-score   | 46.62         | 50.00        | 46.25      | 53.37         |
| chexbert-all_micro avg_f1-score | 48.64         | 52.21        | 42.71      | 53.74         |
| chexbert-5_macro avg_f1-score   | 42.33         | 47.78        | 43.01      | 49.65         |
| chexbert-all_macro avg_f1-score | 33.45         | 35.49        | 33.27      | 38.60         |
| **mimic-cxr-chexpert-findings** |               |              |            |               |
| ROUGEL                          | 29.34         | 23.38        | 25.72      | 26.08         |
| bertscore                       | 57.68         | 50.43        | 54.16      | 51.92         |
| radgraph_simple                 | 34.29         | 31.43        | 28.57      | 30.92         |
| radgraph_partial                | 31.88         | 27.70        | 26.03      | 28.07         |
| radgraph_complete               | 25.24         | 20.13        | 19.67      | 21.15         |
| BLEU                            | 6.51          | 5.60         | 5.38       | 6.26          |
| chexbert-5_micro avg_f1-score   | 54.39         | 53.16        | 55.80      | 46.85         |
| chexbert-all_micro avg_f1-score | 49.41         | 55.95        | 49.88      | 52.72         |
| chexbert-5_macro avg_f1-score   | 46.35         | 49.46        | 46.71      | 41.45         |
| chexbert-all_macro avg_f1-score | 33.29         | 36.61        | 33.47      | 34.29         |
| **mimic-cxr-chexpert-impression** |             |              |            |               |
| ROUGEL                          | 30.65         | 27.33        | 23.68      | 26.44         |
| bertscore                       | 51.49         | 52.91        | 46.47      | 53.32         |
| radgraph_simple                 | 32.64         | 28.44        | 27.12      | 27.33         |
| radgraph_partial                | 30.07         | 25.77        | 23.93      | 24.57         |
| radgraph_complete               | 23.83         | 19.43        | 18.73      | 18.65         |
| BLEU                            | 3.54          | 6.66         | 2.96       | 6.55          |
| chexbert-5_micro avg_f1-score   | 51.89         | 47.58        | 52.03      | 53.37         |
| chexbert-all_micro avg_f1-score | 53.64         | 52.20        | 47.43      | 53.74         |
| chexbert-5_macro avg_f1-score   | 45.31         | 44.62        | 43.96      | 49.65         |
| chexbert-all_macro avg_f1-score | 34.96         | 34.99        | 33.81      | 38.60         |


