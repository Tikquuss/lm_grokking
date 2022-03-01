
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transformers import AutoTokenizer, DataCollatorForLanguageModeling #,DataCollatorWithPadding 
from datasets import load_dataset, arrow_dataset, Dataset, DatasetDict

from loguru import logger
from typing import List, Union, Dict
from functools import partial
import pandas as pd
import tqdm
import random

from .utils import get_extension

# https://huggingface.co/docs/datasets/loading_datasets.html
extensions = {'.csv' : 'csv', '.json' : 'json', '.txt' : 'text'}

def mlm_collate_fn(features, mask_token_id : int, mlm_collator : DataCollatorForLanguageModeling, attn_pad_token_id = 0):
    #keys = [k for k in features[0].keys() if k != label_column]
    keys = features[0].keys()
    L = len(features)
    features = {k : [features[i][k] for i in range(L)] for k in keys}
    x_mlm = mlm_collator(features["input_ids"])
    try :
        features["attention_mask"] = torch.FloatTensor(features["attention_mask"])
        features["token_type_ids"] = torch.LongTensor(features["token_type_ids"])
    except ValueError: #expected sequence of length i at dim 1 (got j)
        #https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L922
        # TODO : avoid using for
        features["token_type_ids"] = pad_sequence(
            [torch.LongTensor(x_i) for x_i in features["token_type_ids"]], 
            padding_value=0,
            batch_first=True
        )
        # TODO : avoid using for, check the good option
        if False :
            padding_mask = pad_sequence(
                [torch.tensor(a_m).int() for a_m in features["attention_mask"]], 
                padding_value=attn_pad_token_id,
                batch_first=True
            )
            mlm_mask = (x_mlm['input_ids'] != mask_token_id).int() # 1 if token == mask_token_id else 0
            features["attention_mask"] = (mlm_mask & padding_mask).float()
        else :
            padding_mask = pad_sequence(
                [torch.FloatTensor(a_m) for a_m in features["attention_mask"]], 
                padding_value=attn_pad_token_id,
                batch_first=True
            )
            features["attention_mask"] = padding_mask 
    
    features['input_ids'] = torch.LongTensor(x_mlm['input_ids'])
    features['labels'] = torch.LongTensor(x_mlm['labels']) 
    #features["mask_token_index"] = torch.where(features['input_ids'] == mask_token_id)[1]
    # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/data/data_collator.py#L767
    features["mask_token_index"] = features['labels'] != -100
    
    return features

def clm_collate_fn(features, pad_token_id : int, attn_pad_token_id : int = 0):
    #keys = [k for k in features[0].keys() if k != label_column]
    keys = features[0].keys()
    L = len(features)
    features = {k : [features[i][k] for i in range(L)] for k in keys}
    """
    # https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L552
    input_ids : torch.LongTensor
    # https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L570
    attention_mask : torch.FloatTensor
    """
    try :
        features["attention_mask"] = torch.FloatTensor(features["attention_mask"])
        features["input_ids"] = torch.LongTensor(features["input_ids"])
    except ValueError: #expected sequence of length i at dim 1 (got j)
        #padding_collator = DataCollatorWithPadding(tokenizer, padding = True, max_length=512)
        #features = padding_collator(features)
        #https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L922
        # TODO : avoid using for
        features["input_ids"] = pad_sequence(
            [torch.LongTensor(x_i) for x_i in features["input_ids"]], 
            padding_value=pad_token_id,
            batch_first=True
        )
        features["attention_mask"] = pad_sequence(
            [torch.FloatTensor(a_m) for a_m in features["attention_mask"]], 
            padding_value=attn_pad_token_id,
            batch_first=True
        )
    #The labels is the same as the inputs, shifted to the left.
    #We duplicate the inputs for our labels. This is because the model of the 🤗 Transformers library 
    #apply the shifting to the right, so we don't need to do it manually.
    #features["labels"] = features["input_ids"].copy()
    features["labels"] = features["input_ids"].clone()
    #features["mask_token_index"] = None

    return features

class LMLightningDataModule(pl.LightningDataModule):
    """Language Modeling (CLM and MLM) dataset"""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        batch_size: int,
        num_workers: int,
        max_length: int,
        dataset_path : Union[str, Dict[str, str]],
        dataset_name : str = "tmp",
        split : str = None,
        num_proc : int = 4,
        text_column : str = 'text',
        label_column : str = None,
        group_texts : bool = True,
        clm : bool = False,
        mlm : bool = False, 
        mlm_probability : float=0.15,
        max_train_samples : int = None,
        max_validation_samples : int = None,
        max_test_samples : int = None
    ):
        """
        Params :
        - tokenizer (AutoTokenizer) : tokenizer
        - batch_size (int) : Number of sentences per batch
        - num_workers (int) : Number of worker processes for DataLoader
        - max_length (int) : Maximum length of sentences
        - dataset_path (str or dict) : 
            * Dict[str, str] : {"train" : "file1,file2,..", "validation" : "...", ...}
            * str : wikitext, oscar ...
            * str : CSV files (with the csv script), JSON files (with the json script), 
                    text files (read as a line-by-line dataset with the text script), parquet files (with the parquet script),
                    pandas pickled dataframe (with the pandas script).
        - dataset_name (str, default='tmp') : 
            * For wikitext : wikitext-103-v1, wikitext-2-v1, wikitext-103-raw-v1, wikitext-2-raw-v1
            * For oscar : unshuffled_deduplicated_af, unshuffled_deduplicated_als ... 
        - split (str, default=None) : train, validation, test...
        - num_proc (int, default = 4) : Number of worker processes for data processing
        - text_column (str, default = 'text') : for csv only, text column
        - label_column (str, default = None) : for csv only, and classification (label colum)
        - group_texts (bool, default=True) : If True, all the documents will be grouped, then divided into blocks of the same length. 
        - clm (bool, default = False) : causal language modeling
        - mlm (bool, default = False) : mask language modeling
        - mlm_probability (float, default=0.15) : fraction of words for which we need to make a prediction (mlm only)
        - max_train_samples (int, default=None) : number of training samples
        - max_validation_samples (int, default=None) : number of validation samples
        - max_test_samples (int, default=None) : number of test samples
        """
        super(LMLightningDataModule, self).__init__()
        assert clm ^ mlm
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.num_proc = num_proc
        self.text_column = text_column
        self.label_column = label_column
        self.group_texts = group_texts
        self.max_train_samples = max_train_samples
        self.max_validation_samples = max_validation_samples
        self.max_test_samples = max_test_samples
    
        if clm :
            #self.collate_fn = None  
            if not self.tokenizer.pad_token_id : self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            self.collate_fn = partial(clm_collate_fn, pad_token_id = self.tokenizer.pad_token_id)  
        if mlm :
            mlm_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability
            )
            self.collate_fn = partial(mlm_collate_fn, mask_token_id=self.tokenizer.mask_token_id, mlm_collator = mlm_collator, attn_pad_token_id = 0)

        self.prepare_data()

    def prepare_data(self):
        """Preparing the dataset"""
        logger.info(f"Dataset {self.dataset_name} loading....")
        # #### TODO
        # train_ds, test_ds = load_dataset(path = self.dataset_path, split=['train[:5000]', 'test[:2000]'])
        # splits = train_ds.train_test_split(test_size=0.1)
        # train_ds = splits['train']
        # val_ds = splits['test']
        # #### 
        if type(self.dataset_path) == str :
            #https://github.com/huggingface/datasets/blob/master/src/datasets/load.py#L1503
            dataset = load_dataset(path = self.dataset_path, name=self.dataset_name, split=self.split)
            #first_key = next(iter(dataset))
            dataset_column_names = dataset.column_names if self.split else dataset.column_names[next(iter(dataset))] 
        elif type(self.dataset_path) == dict :
            #assert len(self.dataset_paths) == 3
            #data_files = {k : v for v, k in zip(self.dataset_paths.values(), ["train", 'validation', 'test'])}
            data_files = {k : v.split(',') for k, v in self.dataset_path.items()}
            assert all([k in ["train", "validation", "test"] for k in data_files.keys()])
            #path = os.path.abspath(os.getcwd())
            first_key = next(iter(data_files))
            ext = get_extension(data_files[first_key][0])
            assert all([get_extension(v_i) == ext for v in data_files.values() for v_i in v])
            path = extensions[ext]
            try :
                dataset = load_dataset( # TODO : see also load_from_disk, Dataset.from_dict
                    path = path, 
                    name = self.dataset_name, 
                    data_files = data_files
                )
                dataset_column_names = dataset.column_names[next(iter(dataset))]
            except ValueError as ve: # Couldn't cast, because column names don't match
                if path == "csv" :
                    """dataset = Dataset.from_dict({
                        k : self.load_from_csv(v, self.text_column, self.label_column, if_shuffle=True, n_samples = None) for k, v in data_files.items()
                    })"""
                    dataset = DatasetDict({
                        k : Dataset.from_dict(self.load_from_csv(v, self.text_column, self.label_column, if_shuffle=True, n_samples = None))
                        for k, v in data_files.items()
                    })
                    dataset_column_names = [] 
                    for k in data_files :
                        dataset_column_names.extend(dataset.column_names[k])
                else :
                    raise ve

            if self.split : dataset = dataset[self.split]
            
        logger.info(f"Loading of {self.dataset_name} datasets completed.")
        
        to_remove_column = [x for x in dataset_column_names if x != self.text_column and x != self.label_column]
        dataset = dataset.remove_columns(to_remove_column)

        if self.group_texts :
            logger.info(f"Tokenize ...")
            def tokenize_function(examples):
                return self.tokenizer(examples[self.text_column])
                
            dataset = dataset.map(tokenize_function, batched=True, num_proc=self.num_proc, remove_columns=[self.text_column])

            max_seq_length = self.max_length
            logger.info(f"Grouping text in block of size {max_seq_length}...")
            def group_texts_fn(examples):
                # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py#L410
                # Concatenate all texts.
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                    # customize this part to your needs.
                total_length = (total_length // max_seq_length) * max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            dataset = dataset.map(group_texts_fn, batched=True, batch_size=self.batch_size, num_proc=self.num_proc)
        else :
            logger.info(f"Tokenize ...")
            def tokenize_function(examples):
                return self.tokenizer(examples[self.text_column], padding=True, truncation=True, max_length=self.max_length)
            dataset = dataset.map(tokenize_function, batched=True, num_proc=self.num_proc, remove_columns=[self.text_column])

        if self.split :
            for attr_name in ["train", "validation", "test"]:
                setattr(self, attr_name, dataset if self.split == attr_name else None)
        else : 
            for attr_name in ["train", "validation", "test"]:
                setattr(self, attr_name, dataset.get(attr_name, None))

        # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py#L442
        """
        if self.max_train_samples is not None and self.train:
            self.train = self.train.select(range(self.max_train_samples))
        if self.max_validation_samples is not None and self.validation:
            self.validation = self.validation.select(range(self.max_validation_samples))
        if self.max_test_samples is not None and self.test :
            self.test = self.test.select(range(self.max_test_samples))
        """
        logger.info(f"Selecting samples...")
        for attr_name in ["train", "validation", "test"]:
            max_samples = getattr(self, "max_%s_samples"%attr_name)
            attr = getattr(self, attr_name)
            if max_samples is not None and attr :
                setattr(self, attr_name, attr.select(range(max_samples)))

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]: 
        return self.test_dataloader()

    @staticmethod
    def load_from_csv(file_list, text_column, label_column = None, if_shuffle=True, n_samples = None):
        data, labels = [], []
        for _index in range(len(file_list)):
            file_item = file_list[_index]
            df = pd.read_csv(file_item)
            for row in tqdm.tqdm(list(df.iterrows()), desc="%s" % file_item):
                row = row[1]
                data.append(row[text_column].strip())
                labels.append(row.get(label_column, None))

        if if_shuffle:
            index = list(range(len(data)))
            random.shuffle(index)
            data = [data[i] for i in index]
            labels = [labels[i] for i in index]

        if label_column is None : return {text_column : data[:n_samples]}
        else : return {text_column : data[:n_samples], label_column : labels[:n_samples]}
