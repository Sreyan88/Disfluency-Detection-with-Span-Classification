# encoding: utf-8


import os
from pytorch_lightning import Trainer
from collections import namedtuple

# from trainer_spanPred import BertLabeling # old evaluation version
# from trainer_spanPred_newEval import BertLabeling # new evaluation version
from trainer import BertNerTagger # start 0111

def evaluate(ckpt, hparams_file):
	"""main"""

	trainer = Trainer(gpus=[2], distributed_backend="dp")
	# trainer = Trainer(distributed_backend="dp")

	model = BertNerTagger.load_from_checkpoint(
		checkpoint_path=ckpt,
		hparams_file=hparams_file,
		map_location=None,
		batch_size=1,
		max_length=128,
		workers=0
	)
	trainer.test(model=model)


if __name__ == '__main__':

	root_dir1 = "/train_logs/"
	midpath = "spanner_bert-large-uncased_spMLen_usePruneTrue_useSpLenTrue_useSpMorphTrue_SpWtTrue_value0.5_96706325"
	model_names = ["epoch=25.ckpt"]
	for mn in model_names:
		print("model-name: ", mn)
		CHECKPOINTS = "Final_english_spanner_final/" + midpath + "/" + mn
		HPARAMS = "Final_english_spanner_final/" + midpath + "/lightning_logs/version_0/hparams.yaml"
		evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)



