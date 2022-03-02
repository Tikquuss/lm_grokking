import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM

from argparse import ArgumentParser
import os
from loguru import logger
import tqdm

from .dataset import LMLightningDataModule
from .language_modelling import LMLightningModule
from .tokenizing import load_tokenizer,  build_tokenizer_from_vocab
from .utils import bool_flag, str2dic, str2dic_int, str2dic_all, to_none, intorstr
from .customized_transformers import Custom2HuggingFace

MODELS_CLASS = {
    "clm" : AutoModelForCausalLM,
    "mlm" : AutoModelForMaskedLM
}

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = ArgumentParser(description="Language Modeling")

    # Main parameters
    parser.add_argument("--model_name", type=str, help="gpt2, bert-base-uncased, roberta-base, facebook/bart-large ...") # toke, model
    parser.add_argument("--from_pretrained", type=bool_flag, help="load a pre-trained model or not") # model
    parser.add_argument("--task", choices=["mlm", "clm"], help="Mask or Causal Language Modeling \
        - Auto-regressive language generation models : GPT2, XLNet, OpenAi-GPT, CTRL, TransfoXL, XLM, Bart, T5 ... (https://huggingface.co/models?filter=causal-lm)... \
        - Mask language model ... (https://huggingface.co/models?filter=masked-lm)") # data, toke, model
    parser.add_argument("--log_dir", type=str, help="Experiment dump path") # trainer
    parser.add_argument("--tokenizer_params", type=str2dic, default="",
                        help="""tokenizer params : 
                            * tokenizer_folder=...,t_class=...,t_type=...,..
                            * vocab_file=...,t_class=...
                        - t_class : roberta_tokenizer, roberta_tokenizer_fast, bert_tokenizer, bert_tokenizer_fast, 
                                    gpt2_tokenizer, gpt2_tokenizer_fast, albert_tokenizer, albert_tokenizer_fast ...
                        - t_type : byte_level_bpe, bert_word_piece, gpt2_bpe, albert_unigram_model ...""") # toke

    # Tokenizer & Dataset 
    parser.add_argument("--dataset_path", type=to_none, default="", 
                        help="""wikitext, oscar ...
                            ```
                            >>> from datasets import list_datasets
                            >>> datasets_list = list_datasets()""") # data
    parser.add_argument("--dataset_name", type=to_none, default="tmp", 
                        help="""
                            - For wikitext : wikitext-103-v1, wikitext-2-v1, wikitext-103-raw-v1, wikitext-2-raw-v1
                            - For oscar : unshuffled_deduplicated_af, unshuffled_deduplicated_als ... 
                            - ...""") # data
    parser.add_argument("--train_data_files", type=to_none, default="", help="path_to_file1,path_to_file2,...") # data
    parser.add_argument("--validation_data_files", type=to_none, default="", help="path_to_file1,path_to_file2,...") # data
    parser.add_argument("--test_data_files", type=to_none, default="", help="path_to_file1,path_to_file2,...")
    parser.add_argument("--split", type=to_none, default="", help="train, validation, test...")
    parser.add_argument("--text_column", type=to_none, default="text", help="For csv only, text column") # data
    parser.add_argument("--label_column", type=to_none, default="", help="For csv only, and classification (label colum)") # data
    parser.add_argument("--group_texts", type=bool_flag, default=True, help="If True, all the documents will be grouped, then divided into blocks of the same length") # data
    parser.add_argument("--max_samples", type=str2dic_int, default="", help="train=100,validation=...test=...") # data
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Fraction of words for which we need to make a prediction (mlm only)") # data
    #parser.add_argument("--word_mask_keep_rand", type=str2dic, default="0.8,0.1,0.1", help="Fraction of words to mask out / keep / randomize, among the words to predict") # data
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch") # data
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data processing and for DataLoader") # data
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of sentences") # toke, data

    # Model
    parser.add_argument("--model_params", default="", type=str2dic_all, help="""
        - Change some parameters of the core model. To see the different changeable parameters, do :
            ```
            >>> model_name = "bert-base-uncased"  # "gpt2", ...  
            >>> config = AutoConfig.from_pretrained(model_name)
            >>> print(config)
            ```
        - Examples :
            - gpt2 : "activation_function=str(gelu_new),attn_pdrop=float(0.1),n_ctx=int(1024),n_embd=int(768),n_head=int(12),n_layer=int(12),\
                      n_positions=int(1024),vocab_size=int(50257),..."
            - bert-base-uncased : "attention_probs_dropout_prob=float(0.1),hidden_act=str(gelu),hidden_dropout_prob=float(0.1),hidden_size=int(768),\
                                   intermediate_size=int(3072),max_position_embeddings=int(512),num_attention_heads=int(12),num_hidden_layers=int(12),\
                                   pad_token_id=int(0),position_embedding_type=str(absolute),vocab_size=int(30522),..."
            
            - custom model : "custom=bool(True),hf_transformer=bool(False),emb_dim=int(768),dim_feedforward=int(3072),n_heads=int(12),n_layers=int(12),\
                              dropout=float(0.1),attention_dropout=float(0.1),n_positions=int(512),gelu_activation=bool(True),\
                              sinusoidal_embeddings=bool(True),share_inout_emb=bool(True),tim_layers_pos=str(2-3-4),n_s=int(2),use_group_comm=bool(True)"
        """) # model
    if parser.parse_known_args()[0].model_params: pass

    ## Optimizer
    parser.add_argument("--optimizer_params", type=str, default="adam,lr=0.00001,beta1=0.9,beta2=0.99,eps=0.00000001", help="""
                - optimizer parameters : adam_inverse_sqrt,lr=0.00001,beta1=0.9,beta2=0.99,eps=0.00000001 ...
                - classes : CustomAdam (custom_adam), Adam (adam), AdamInverseSqrtWithWarmup (adam_inverse_sqrt), AdamCosineWithWarmup (adam_cosine),
                            Adadelta (adadelta), Adagrad (adagrad), Adamax (adamax), ASGD (asgd), SGD (sgd), RMSprop (rmsprop), Rprop (rprop)
                - See https://pytorch.org/docs/stable/optim.html for optimizers parameters
    """) # optim : training
    parser.add_argument("--lr_factor", type=float, default=0.1, help="learning rate scheduler factor") # optim : training
    parser.add_argument("--lr_patience", type=int, default=4, help="learning rate scheduler patience") # optim : training

    ## Training
    parser.add_argument("--validation_metrics", type=str, default="val_loss", help="Validation metrics : val_acc, val_loss, val_ppl ...") # trainer
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epoch") # trainer
    parser.add_argument("--checkpoint_path", type=to_none, default="", help="Reload a checkpoint") # trainer
    #parser.add_argument("--limit_batches", type=str2dic_int, default="train=1.,validation=1.,test=1.", help="") # trainer
    parser.add_argument("--limit_train_batches", type=float, default=1., help="limit batches for training data") # trainer
    parser.add_argument("--limit_val_batches", type=float, default=1., help="limit batches for validation data") # trainer
    parser.add_argument("--limit_test_batches", type=float, default=1., help="limit batches for test data") # trainer
    parser.add_argument("--eval_only", type=bool_flag, default=False, help="Only run evaluations") # trainer
    parser.add_argument("--eval_split", choices=["train", "validation", "test"], default="test", help="evaluation on training/validation/test data") # evaluation
    parser.add_argument("--val_check_interval", type=float, default=0.5, help="How often to check the validation set. Use float to check within a training epoch, \
                                                                                use int to check every n steps (batches)") # trainer
    parser.add_argument("--early_stopping_patience", type=int, default=10, 
                        help="Early stopping patience : If the model does not converge during these numbers of steps, stop the training") # trainer
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulates grads every k batches or as set up in the dict (1 = no accumulation)") # trainer
    parser.add_argument("--save_top_k", type=int, default=1, help="""The best `save_top_k` models according to the quantity monitored will be saved. 
                                    If save_top_k == 0, no models are saved. if save_top_k == -1, all models are saved.""") # trainer
    parser.add_argument("--strategy", type=str, default="ddp", help="ddp (DistributedDataParallel), ddp_spawn ...") # trainer
    parser.add_argument("--auto_scale_batch_size", type=to_none, default=None, # "binsearch" 
                help="Automatically tries to find the largest batch size that fits into memory, before any training") # trainer
    parser.add_argument("--auto_lr_find", type=bool_flag, default=False, help="runs a learning rate finder algorithm") # trainer
    parser.add_argument("--deterministic", type=bool_flag, default=False, help='ensures reproducibility') # trainer
    parser.add_argument("--freeze_transformer", type=bool_flag, default=False, help="Freeze transformer parameters") # model
    parser.add_argument("--accelerator", type=str, default="auto", help="accelerator types : cpu, gpu, tpu, ipu, auto") 
    parser.add_argument("--devices", type=intorstr, default="auto", help="number of cpu processes, of gpu/tpu cores ...")
    parser.add_argument("--random_seed", type=int, default=2021, help="random seed for reproductibility")
    parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=0, help="It is better to set a value > 1 for mlm") 

    # Predict
    parser.add_argument("--predict_params", default="", type=str2dic_all, help="""
        - Prediction parameters : 
            - clm : generate sentences (greedy_search / beam_search / top-k_sampling / top-p_sampling) 
            - mlm : fill mask (mlm)
        - Examples :
            - bert-base-uncased : "a=int(1)"  (there are no parameters to specify, enter anything to activate the prediction)
            - gpt2 : "type=str(greedy_search),max_length=int(50)"
            - gpt2 : "type=str(beam_search),max_length=int(50),num_beams=int(5),early_stopping=bool(True),no_repeat_ngram_size=int(1)"
            - gpt2 : "type=str(top-k_sampling),max_length=int(50),do_sample=bool(True),top_k=int(0)"
            - gpt2 : "type=str(top-k_sampling),max_length=int(50),do_sample=bool(True),top_k=int(0),temperature=float(0.5)"
            - gpt2 : "type=str(top-p_sampling),max_length=int(50),do_sample=bool(True),top_k=int(0),top_p=float(0.92)"
        """) # model

    return parser

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        logger.info("=========== Training is started! ========")
        #logger.info(pl_module.model)
    def on_train_end(self, trainer, pl_module):
        logger.info("======== Training is done! ======== ")

def main(params) :
    # Seed everything (sets seeds for numpy, torch, python.random and PYTHONHASHSEED)
    pl.seed_everything(params.random_seed, workers=True)

    root_dir = os.path.join(params.log_dir, params.task)
    os.makedirs(root_dir, exist_ok=True)

    # Tokenizer
    logger.info("Tokenizer...")
    tokenizer_folder = getattr(params.tokenizer_params, 'tokenizer_folder', None)
    if getattr(params.tokenizer_params, 'vocab_file', None) :
        tokenizer = build_tokenizer_from_vocab(vocab_file=params.tokenizer_params.vocab_file, t_class1 = getattr(params.tokenizer_params, 't_class'))
    else :
        tokenizer = load_tokenizer(
            model_name = params.model_name if not tokenizer_folder else None, 
            tokenizer_folder=tokenizer_folder, 
            t_class = getattr(params.tokenizer_params, 't_class', None), 
            t_type = getattr(params.tokenizer_params, 't_type', None), 
            task = params.task, 
            MAX_LEN = params.max_length,
            vocab_file = None, 
            merges_file = None
        )
    
    # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py#L393
    params.block_size = params.max_length
    if params.block_size is None:
        params.block_size = tokenizer.model_max_length
        if params.block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            params.block_size = 1024
    else:
        if params.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({params.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        params.block_size = min(params.block_size, tokenizer.model_max_length)

    ## Dataset
    logger.info(f"{params.dataset_name} lightning data module creation...")
    if not params.dataset_path :
        params.dataset_path = {k : getattr(params, f"{k}_data_files") for k in ["train", "validation", "test"] if getattr(params, f"{k}_data_files")}
    pl_data_module = LMLightningDataModule(
            tokenizer,
            params.batch_size,
            params.num_workers,
            max_length = params.block_size,
            dataset_path = params.dataset_path,
            dataset_name = params.dataset_name,
            split = params.split,
            num_proc = params.num_workers,
            text_column = params.text_column,
            label_column = params.label_column,
            group_texts = params.group_texts,
            clm = params.task == "clm",
            mlm = params.task == "mlm", 
            mlm_probability = params.mlm_probability,
            max_train_samples = getattr(params.max_samples, 'train', None), 
            max_validation_samples = getattr(params.max_samples, 'validation', None),
            max_test_samples = getattr(params.max_samples, 'test', None)
        )

    ## Model
    logger.info(f"{params.model_name} model building...")

    custom = params.model_params.pop("custom", False) if params.model_params else False
    if not custom :
        if params.from_pretrained :
            model = MODELS_CLASS[params.task].from_pretrained(params.model_name)#, pad_token_id=tokenizer.eos_token_id)
        else :
            config = AutoConfig.from_pretrained(params.model_name)
            for attr_name, attr_val in params.model_params.items() :
                setattr(config, attr_name, attr_val)
            config.vocab_size = tokenizer.vocab_size
            model = MODELS_CLASS[params.task].from_config(config)
        model.resize_token_embeddings(len(tokenizer))
    else :
        """
        params: 
            - *n_words, eos_index, pad_index, *mask_token_id, *mlm_probability
            - emb_dim, dim_feedforward, n_heads, n_layers, dropout, attention_dropout, n_positions, gelu_activation, 
              sinusoidal_embeddings, share_inout_emb, tim_layers_pos, n_s, use_group_comm
        optional params : use_lang_emb (False), n_langs (1), sample_alpha (0)
        """
        model_params = params.model_params
        assert model_params is not None
        model_params.n_words = tokenizer.vocab_size 
        model_params.pad_index = getattr(tokenizer, "pad_token_id", None)
        if model_params.pad_index is None :
            model_params.pad_index = getattr(tokenizer, "eos_token_id", None)
        model_params.eos_index = getattr(tokenizer, "eos_token_id", None) # clm
        if model_params.eos_index is None : # clm
            model_params.eos_index = getattr(tokenizer, "sep_token_id", None) # clm
        model_params.context_size = 0 # clm (0 means that the first elements in sequences won't have any context)
        model_params.mask_token_id = getattr(tokenizer, "mask_token_id", None) # mlm
        model_params.mlm_probability = params.mlm_probability # mlm

        model = Custom2HuggingFace(model_params, task=params.task)
    
    if params.freeze_transformer :
        for param in model.parameters():
            param.requires_grad = False

    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(f"Training new model - Total size={n_params/2**20:.2f}M params")

    ## Trainer  (https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
    logger.info("Trainer building...")
    trainer_config = {
        "max_epochs": params.max_epochs,

        "default_root_dir" : root_dir,
        #"log_every_n_steps" : max(len(pl_data_module.train) // params.batch_size, 0),
        #"weights_save_path" : os.path.join(root_dir, "weights"),
        "auto_scale_batch_size" : params.auto_scale_batch_size, # None
        "auto_select_gpus" : True,
        "auto_lr_find": params.auto_lr_find,
        "benchmark" : False,
        "deterministic" : params.deterministic,
        
        "val_check_interval" : params.val_check_interval,
        "accumulate_grad_batches" : params.accumulate_grad_batches,
        "strategy": params.strategy,
        "limit_train_batches" : params.limit_train_batches, 
        "limit_val_batches" : params.limit_val_batches,
        "limit_test_batches": params.limit_test_batches,

        "accelerator" : params.accelerator,
        "devices" : params.devices,
        "reload_dataloaders_every_n_epochs" : params.reload_dataloaders_every_n_epochs,
        "weights_summary":"full", # "top", None
    }
    if not params.eval_only and not params.predict_params :
        trainer_config["log_every_n_steps"] = max(len(pl_data_module.train) // params.batch_size, 1)
    
    if torch.cuda.is_available():
        trainer_config["gpus"] = -1
    
    early_stopping_callback = EarlyStopping(
        monitor=params.validation_metrics, patience=params.early_stopping_patience, verbose=False, strict=True,
        mode = (lambda s : "min" if 'loss' in s or "ppl" in s else 'max')(params.validation_metrics)
    )
    model_checkpoint_callback = ModelCheckpoint(
            dirpath=root_dir,
            filename="{epoch}-{%s:.4f}"%params.validation_metrics,
            monitor=params.validation_metrics,
            save_top_k=params.save_top_k,
    )
    trainer_config["callbacks"] = [
        early_stopping_callback, 
        model_checkpoint_callback, 
        PrintCallback()
    ]
    
    pl_trainer = pl.Trainer(**trainer_config)

    ## Training / Evaluation / Prediction
    logger.info("Training / Evaluation / Prediction...")
    pl_model = LMLightningModule(
        model=model,
        task=params.task,
        optimizer_params=params.optimizer_params,
        lr_factor=params.lr_factor,
        lr_patience=params.lr_patience,
        decoder_start_token_id=tokenizer.pad_token_id
    )

    if params.predict_params :
        # Generate sentences (clm) / fill mask (mlm)
        assert params.task == "mlm" or getattr(params.predict_params, "max_length", None) is not None
        logg_text = "Generate sentences" if params.task == "clm" else "fill mask"
        logger.info(f"{logg_text} ...")
        if params.eval_split == "train":
            pl_data_module.predict_dataloader = pl_data_module.train_dataloader
        elif params.eval_split == "validation" :
            pl_data_module.predict_dataloader = pl_data_module.val_dataloader
        pl_model.eval()
        
        for attr_name, attr_val in params.predict_params.items() :
            setattr(pl_model, attr_name, attr_val)

        output = pl_trainer.predict(pl_model, datamodule=pl_data_module, ckpt_path=params.checkpoint_path)
        
        params.output_file = getattr(params.predict_params, "output_file", os.path.join(root_dir, "predict.txt"))
        with open(params.output_file, "w") as of:
            keys = list(output[0].keys())  # input_ids, output_ids, ...
            key_0 = keys[0]
            for batch in tqdm.tqdm(output, desc="Save in %s ..." % params.output_file):
                for i in range(len(batch[key_0])) :
                    of.writelines([f"{k} : %s\n"% tokenizer.decode(batch[k][i], skip_special_tokens=False) for k in keys])
                    of.write('\n')

        logger.info(f"{logg_text} completed.")
        exit()

    if not params.eval_only :
        # Tuning
        if params.auto_scale_batch_size or params.auto_lr_find : # find the batch size / learning rate
            logger.info(f"Tuning model...")
            pl_trainer.tune(pl_model, datamodule=pl_data_module)

        # Training
        logger.info("Training starts...")
        pl_model.train()
        pl_trainer.fit(pl_model, datamodule=pl_data_module, ckpt_path=params.checkpoint_path)
        logger.info("Training completed.")
        logger.info("Testing starts....")
        pl_model.eval()
        pl_trainer.test(pl_model, datamodule=pl_data_module)
        logger.info("Testing completed.")
    else :
        # Evaluation
        logger.info("Evaluation starts....")
        if params.eval_split == "train":
            pl_data_module.test_dataloader = pl_data_module.train_dataloader
        elif params.eval_split == "validation" :
            pl_data_module.test_dataloader = pl_data_module.val_dataloader
        pl_model.eval()
        pl_trainer.test(pl_model, datamodule=pl_data_module, ckpt_path=params.checkpoint_path)
        logger.info("Evaluation completed.")

if __name__ == "__main__":

    # generate parser / parse parameters
    #params = get_parser().parse_args()
    parser = get_parser() 
    # To be able to pass all the possible parameters of Trainer (https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
    #parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args()
    # run the experiments
    main(params)
