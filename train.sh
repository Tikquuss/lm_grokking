#!/bin/bash

None="_None_"

## clm : gpt2, facebook/bart-large, ... (https://huggingface.co/models?filter=causal-lm)
## mlm : bert-base-uncased, roberta-base, ... (https://huggingface.co/models?filter=masked-lm)
model_name="gpt2"
task=clm
#model_name="bert-base-uncased"
#task=mlm
reload_dataloaders_every_n_epochs=0
if [ $task="mlm" ]; then
	# it's better to apply masks in a different way in every epoch
	reload_dataloaders_every_n_epochs=1
fi

from_pretrained=True

log_dir="../log_files"

## If from_pretrained
tokenizer_params=$None
## Custom pre-trained tokenized (If trained from a pre-trained model)
#tokenizer_params="tokenizer_folder=/content"
## Custom pre-trained tokenized (If not trained from a pre-trained model)
#tokenizer_params="tokenizer_folder=/content,t_class=bert_tokenizer_fast,t_type=bert_word_piece"
## ... see tokenizing.py

## Online dataset
dataset_path="wikitext"
dataset_name="wikitext-2-raw-v1"
train_data_files=$None
validation_data_files=$None
test_data_files=$None
## Local dataset
#dataset_path=$None
#dataset_name="tmp"
#datapath=/content
#train_data_files=${datapath}/data_train.csv
#validation_data_files=${datapath}/data_val.csv
#test_data_files=${datapath}/data_test.csv
## In case of prediction or evaluation, it is possible to put the data split not using to None (train_data_files=$None)

## For csv dataset, set the good columns name (label_column is for text classification fine-tuning, comming soon)
text_column=text
label_column=$None
group_texts=True

## use a small part of dataset?
max_samples=$None
#max_samples="train=800"
#max_samples="train=800,validation=100,test=100"

## Batch size and max length (make sure max_length < n_positions/max_position_embeddings of the model)
batch_size=32
max_length=512

# If from_pretrained
model_params=$None
# For gpt2
#model_params="n_ctx=int(1024),n_embd=int(768),n_head=int(12),n_layer=int(12),n_positions=int(${max_length})"
# For bert
#model_params="hidden_size=int(768),intermediate_size=int(3072),max_position_embeddings=int(${max_length}),num_attention_heads=int(12),num_hidden_layers=int(12)"
# ...
# For custom implementation
#model_params="custom=bool(True),hf_transformer=bool(False),emb_dim=int(768),dim_feedforward=int(3072),n_heads=int(12),n_layers=int(12)"
#model_params="${model_params},dropout=float(0.1),attention_dropout=float(0.1),n_positions=int(${max_length}),gelu_activation=bool(True)"
#model_params="${model_params},sinusoidal_embeddings=bool(True),share_inout_emb=bool(True),tim_layers_pos=str(2-3-4),n_s=int(2),use_group_comm=bool(True)"

validation_metrics=val_loss
max_epochs=10

## Replace if necessary (continue training a model, evaluate a model ...)
checkpoint_path=$None
#checkpoint_path=/content/log_files/mlm/epoch=1-val_loss=2.0445.ckpt

## Evaluate on train / validation / test data (test data by default)
eval_only=False
#eval_split="train"
#eval_split="validation"
eval_split="test"

## distributed data parallel strategy
#strategy="ddp_spawn"
strategy="ddp"

auto_scale_batch_size=$None
#auto_scale_batch_size=binsearc
auto_lr_find=False

# gpu, cpu, tpu, ipu, ... auto
accelerator="gpu"
#accelerator="cpu"

## Make prediction (text generation / fill mask)
predict_params=$None
## fill mask
#predict_params="a=int(1)"
## text generation
#predict_params="type=str(greedy_search),max_length=int(50)"
## ... see trainer.py

## Intrinsic Dimension estimation
ID_params=$None
#ID_params="method=str(twonn)"
#ID_params="method=str(mle),k=int(2),averaging_of_inverses=bool(True)"

python3 -m src.trainer \
		--model_name $model_name \
		--from_pretrained $from_pretrained \
		--task $task \
		--log_dir $log_dir \
		--tokenizer_params $tokenizer_params \
		--dataset_path $dataset_path \
		--dataset_name $dataset_name \
		--train_data_files $train_data_files \
		--validation_data_files $validation_data_files \
		--test_data_files $test_data_files \
		--split $None \
		--text_column $text_column \
		--label_column $label_column \
		--group_texts $group_texts \
		--max_samples $max_samples \
		--mlm_probability 0.15 \
		--batch_size $batch_size \
		--num_workers 4 \
		--max_length $max_length \
		--model_params $model_params \
		--optimizer_params adam,lr=0.00001,beta1=0.9,beta2=0.99,eps=0.00000001 \
		--lr_factor 0.1 \
		--lr_patience 4 \
		--validation_metrics $validation_metrics \
		--max_epochs $max_epochs \
		--checkpoint_path $checkpoint_path \
		--limit_train_batches 1.0 \
		--limit_val_batches 1.0 \
		--limit_test_batches 1.0 \
		--eval_only $eval_only \
		--eval_split $eval_split \
		--val_check_interval 0.5 \
		--early_stopping_patience 10 \
		--accumulate_grad_batches 1 \
		--save_top_k 1 \
		--strategy $strategy \
		--auto_scale_batch_size $auto_scale_batch_size \
		--auto_lr_find $auto_lr_find \
		--deterministic False \
		--freeze_transformer False \
		--accelerator $accelerator \
		--devices auto \
		--random_seed 2021 \
		--reload_dataloaders_every_n_epochs $reload_dataloaders_every_n_epochs \
		--predict_params $predict_params \
		--ID_params $ID_params