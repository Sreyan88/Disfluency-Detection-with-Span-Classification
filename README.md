# README for Disfluency-Detection-with-Span-Classification
This repository contains the implementation of the paper: "[**Span Classification with Structured Information for Disfluency Detection in Spoken Utterances**](https://arxiv.org/pdf/2203.16028.pdf)".
****
##  Abstract
Existing approaches in disfluency detection focus on solving a token-level classification task for identifying and removing disfluencies in text. Moreover, most works focus on leveraging only contextual information captured by the linear sequences in text, thus ignoring the structured information in text which is efficiently captured by dependency trees. In this paper, building on the span classification paradigm of entity recognition, we propose a novel architecture for detecting disfluencies in
transcripts from spoken utterances, incorporating both contextual information through transformers and long-distance structured information captured by dependency trees, through graph convolutional networks (GCNs). Experimental results show that our proposed model achieves state-of-the-art results on the widely used English Switchboard for disfluency detection and outperforms prior-art by a significant margin.
****
## Proposed model
![Alt text](pic/model.png?raw=true "Proposed Architecture")
****
## Requirements
All the dependencies are mentioned in requirements.txt file and can be installed using the following command
```python
pip install -r requirements.txt
```
****
## Data Preprocessing
The dataset to be used for this model needs to be preprocessed before feeding it to the model. To do the same we provide a `dataprocess/bio2spannerformat.py`. First, you need to download datasets, and then convert them into BIO2 tagging format. We have used the switchboard dataset. Note that we use the dependency head index to incorporate the structured information.
## How to Run?
To run the experiment download and extract the pretrained model in the root directory, we use [BERT-Large](https://github.com/google-research/bert). You may need to change the `DATA_DIR`, `PRETRAINED`, `dataname`, `n_class` to your own dataset path, pre-trained model path, dataset name, and the number of labels in the dataset, respectively.

```
./run.sh
```
