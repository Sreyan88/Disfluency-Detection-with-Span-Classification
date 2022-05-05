import flair
import torch
import argparse
import os
os.environ['CUDA_VISIBLE-DEVICES']="1" 

parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input', '-i',
                        help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output', '-o',
                        help='Name of the output folder')
parser.add_argument('--gpu', '-g',
                        help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')


args = parser.parse_args()
input_folder=args.input
output_folder=args.output
gpu_type=args.gpu


flair.device = torch.device(gpu_type)
from typing import List
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import *
from transformers import LukeModel
# from flair.embeddings import TransformerWordEmbeddings

# Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text', 1:'ner'}

data_folder = input_folder

tag_type = 'ner'

corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='train.txt',
                              dev_file='dev.txt',test_file='test.txt',column_delimiter="\t", comment_symbol="# id")

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

embedding_types: List[TokenEmbeddings] = [
    #  TransformerWordEmbeddings('studio-ousia/luke-large',fine_tune = True,model_max_length=512, allow_long_sentences=False),
    TransformerWordEmbeddings('bert-base-uncased',fine_tune = True,model_max_length=512),
 ]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# trainer.train(output_folder, learning_rate=0.01,
#               mini_batch_size=64,
#               max_epochs=150)
#from pathlib import Path

# Load from checkpoint
# checkpoint = 'output_bert/best-model.pt'
# trained_model = SequenceTagger.load(checkpoint)


# trainer.resume(trained_model,base_path=output_folder, learning_rate=0.01,max_epochs=100, mini_batch_size=32,embeddings_storage_mode='gpu',main_evaluation_metric=('macro avg', 'f1-score'))

trainer.train(output_folder, learning_rate=0.01,
             mini_batch_size=16,
             max_epochs=50,embeddings_storage_mode='gpu',main_evaluation_metric=('macro avg', 'f1-score'))

# trainer.final_test('output_english_luke_large', eval_mini_batch_size=16, main_evaluation_metric=('macro avg', 'f1-score'))
