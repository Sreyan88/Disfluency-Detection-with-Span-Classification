# Disfluency-Detection-with-Span-Classification
This repository contains the implementation of the paper: "[**Span Classification with Structured Information for Disfluency Detection in Spoken Utterances**](https://arxiv.org/pdf/2203.16028.pdf)".
****

##  Abstract
Existing approaches in disfluency detection focus on solving a token-level classification task for identifying and removing disfluencies in text. Moreover, most works focus on leveraging only contextual information captured by the linear sequences in text, thus ignoring the structured information in text which is efficiently captured by dependency trees. In this paper, building on the span classification paradigm of entity recognition, we propose a novel architecture for detecting disfluencies in
transcripts from spoken utterances, incorporating both contextual information through transformers and long-distance structured information captured by dependency trees, through graph convolutional networks (GCNs). Experimental results show that our proposed model achieves state-of-the-art results on the widely used English Switchboard for disfluency detection and outperforms prior-art by a significant margin.
****

## Proposed Model Architecture

<p align="center">
<img src="assets/model.png">
</p>

## Requirements
All the dependencies are mentioned in requirements.txt file and can be installed using the following command:  

```
pip install -r requirements.txt
```
****
### Data
For Data related information please refer the information provided in the `data` directory.

### Data Preprocessing
The dataset to be used for this model needs to be preprocessed before feeding it to the model. To do the same we provide a `dataprocess/bio2spannerformat.py`. First, you need to download datasets, and then convert them into BIO2 tagging format. We have used the switchboard dataset. Note that we use the dependency head index to incorporate the structured information. The data format can be understood by the `dummy` data provided in the `data` directory. `english_bio` is the data in the BIO2 format while `english` is the preprocessed data.  

### How to Run?
To run the experiment download and extract the pretrained model in the root directory. For our experiments we use the [BERT-Large Architecture](https://github.com/google-research/bert). You may need to change the `DATA_DIR`, `PRETRAINED`, `dataname`, `n_class` to your own dataset path, pre-trained model path, dataset name, and the number of labels in the dataset, respectively.
`Note: We provide a dummy dataset in the data directory, as Switchboard dataset that we have used is not an opensource dataset. The dummy data is not a part of the switchboard dataset.`

```
./run.sh
```
Pretrained models can be downloaded from this [**link**](https://drive.google.com/file/d/1jb30JrqxYA7hWS0nk6_N9pR1X1WO7qha/view?usp=sharing)

For running the token classification task use the `flair_train.py` by using the following command. 
```
python flair_train.py --input input --output output --gpu cuda:1
```
The input data must be in BIO or IO format in the `input` directory to run this task.

### Evaluate
To calculate the score, use the `calculate_score.py` which takes `gold` and `prediction` files as input.

***

Our code is inspired by [**SpanNER**](https://github.com/neulab/SpanNER) and [**SynLSTM-for-NER**](https://github.com/xuuuluuu/SynLSTM-for-NER)
