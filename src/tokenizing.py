import torch

from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, RobertaTokenizerFast, BertTokenizerFast, GPT2TokenizerFast, AlbertTokenizerFast
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, AlbertTokenizer
from tokenizers.processors import BertProcessing
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from datasets import load_dataset

import argparse
import os

from .utils import bool_flag, dir_path, file_path, csv2txt
from .utils import MASK_WORD, SEP_WORD, CLS_WORD, PAD_WORD, UNK_WORD, special_tokens

TOKENIZERS_TYPE = {
    "byte_level_bpe" : ByteLevelBPETokenizer,
    "bert_word_piece" : None, # BERT
    "gpt2_bpe" : None, # GPT2
    "albert_unigram_model": None, # Albert, T5
}

TOKENIZERS_CLASS = {
    "roberta_tokenizer" : RobertaTokenizer,
    "roberta_tokenizer_fast": RobertaTokenizerFast, 
    "bert_tokenizer" : BertTokenizer,
    "bert_tokenizer_fast" : BertTokenizerFast,
    "gpt2_tokenizer" : GPT2Tokenizer, 
    "gpt2_tokenizer_fast": GPT2TokenizerFast,
    "albert_tokenizer" : AlbertTokenizer,
    "albert_tokenizer_fast": AlbertTokenizerFast
}

common_doc = """
        - args.files / args.path (str) : files containing your datasets / name of your dataset (wikitext, ...)
        - dataset_name (str, default = "tmp") : name of the dataset (if args.path is not None, see the documentation)
        - args.model_name (str) : gpt2, bert-base-uncased, ...
        - args.vocab_size (int) : size of the vocabulary
        - args.save_to (str) : where to save the tokenizer after training
        - args.split (str, defaut = 'train') : train, validation, test
        - args.batch_size (int) : batch size"""

common_doc_with_min_freq = """
        %s
        - args.min_frequency (int) : if #{word, corpus} > min_frequency, vocab.append(word)"""%common_doc

common_doc_with_show_progress = f"""
        {common_doc_with_min_freq}
        - args.show_progress (bool) : show progress during training or not?"""

def batch_iterator(dataset, batch_size, text_column):
    """To avoid loading everything into memory, we define a Python iterator."""
    #return [dataset[i : i + batch_size]["text"] for i in range(0, len(dataset), batch_size)]
    for i in range(0, len(dataset), batch_size):
        #yield dataset[i : i + batch_size][text_column]
        yield dataset[i : i + batch_size]["text"]

def train_from_existing(args) :
    f"""
    If you want to train a tokenizer with the exact same algorithms and parameters as an existing one (gpt2, bert-base-uncased, ...)
    {common_doc}
    How to use?
        import torch
        tokenizer = torch.load(os.path.join(args.save_to, "tokenizer.pt"))
    """
    if args.path :
        dataset = load_dataset(path = args.path, name=args.dataset_name, split=args.split)
    else : 
        """       
        if args.split : 
            dataset = load_dataset(path = os.path.abspath(os.getcwd()), name=args.dataset_name, data_files={args.split : args.files})
            dataset = dataset[args.split]
        else :
            dataset = load_dataset(path = os.path.abspath(os.getcwd()), name=args.dataset_name, data_files=args.files)
        """
        dataset = load_dataset(path = os.path.abspath(os.getcwd()), name=args.dataset_name, data_files={args.split : args.files})
        dataset = dataset[args.split]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    assert tokenizer.is_fast
    tokenizer = tokenizer.train_new_from_iterator(batch_iterator(dataset, args.batch_size, args.text_column), args.vocab_size, new_special_tokens=special_tokens)
    #tokenizer.model.save(args.save_to)
    #tokenizer.save_pretrained("tokenizer_pt")
    torch.save(tokenizer, os.path.join(args.save_to, "tokenizer.pt"))

def train_from_iterator(tokenizer, args, trainer=None):
    """ Train a given tokenizer from data iterator"""
    if args.path :
        dataset = load_dataset(path = args.path, name=args.dataset_name, split=args.split)
    else :
        dataset = load_dataset(path = os.path.abspath(os.getcwd()), name=args.dataset_name, data_files={args.split : args.files})
        dataset = dataset[args.split]
    if trainer is not None :
        tokenizer.train_from_iterator(batch_iterator(dataset, args.batch_size, args.text_column), trainer = trainer)
    else :
        tokenizer.train_from_iterator(batch_iterator(dataset, args.batch_size, args.text_column), vocab_size=args.vocab_size, min_frequency=args.min_frequency,
                                            show_progress=True, special_tokens=special_tokens)
    return tokenizer

def general(args) :
    f"""
    Customize tokenizer training
        - args.type (str) : byte_level_bpe, ...
        {common_doc_with_min_freq}
    How to use?
        import torch
        tokenizer = torch.load(os.path.join(args.save_to, "tokenizer.pt"))
    """
    tokenizer = TOKENIZERS_TYPE[args.type](lowercase=args.lowercase)
    # Customize training
    if args.path or args.split :
        tokenizer = train_from_iterator(tokenizer, args)
    else :
        tokenizer.train(
            files = args.files,
            vocab_size=args.vocab_size, 
            min_frequency=args.min_frequency,
            show_progress=True,
            special_tokens=special_tokens
        )
    #Save the Tokenizer to disk
    tokenizer.save_model(args.save_to)
    #tokenizer.save_pretrained("tokenizer_pt")

def bert_word_piece(args) :
    f"""
    Customize tokenizer training : WordPiece model like BERT
        {common_doc_with_show_progress}
    How to use?
        import torch
        from transformers import BertTokenizerFast # or BertTokenizer
        tokenizer = torch.load(os.path.join(args.save_to, "tokenizer.pt"))
        tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
    """
    tokenizer = Tokenizer(models.WordPiece(unl_token=UNK_WORD))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=args.lowercase)
    # lower case, apply NFD normalization and strip the accents
    #tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    # BertPreTokenizer pre-tokenizes using white space and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(
        vocab_size=args.vocab_size, 
        min_frequency=args.min_frequency, 
        show_progress=args.show_progress, 
        special_tokens=special_tokens
    )
    if args.path or args.split :
        tokenizer = train_from_iterator(tokenizer, args, trainer=trainer)
    else :
        tokenizer.train(files = args.files, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id(CLS_WORD)),
            ("[SEP]", tokenizer.token_to_id(SEP_WORD)),
        ],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.model.save(args.save_to)
    torch.save(tokenizer, os.path.join(args.save_to, "tokenizer.pt"))
    
def gpt2_bpe(args) :
    f"""
    Customize tokenizer training : BPE model like GPT-2
        {common_doc_with_show_progress}
    How to use?
        import torch
        from transformers import GPT2TokenizerFast # or GPT2Tokenizer
        tokenizer = torch.load(os.path.join(args.save_to, "tokenizer.pt"))
        tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    """
    tokenizer = Tokenizer(models.BPE(lowercase=args.lowercase))
    # byte level pre-tokenize
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, 
        min_frequency=args.min_frequency, 
        show_progress=args.show_progress, 
        special_tokens=["<|endoftext|>"]#+special_tokens
    )
    if args.path or args.split :
        tokenizer = train_from_iterator(tokenizer, args, trainer=trainer)
    else :
        tokenizer.train(files = args.files, trainer=trainer)
    # post-processor and decoder
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.model.save(args.save_to)
    torch.save(tokenizer, os.path.join(args.save_to, "tokenizer.pt"))
    
def albert_unigram_model(args) :
    f"""
    Customize tokenizer training : Unigram model like Albert, T5
        {common_doc_with_show_progress}
    How to use?
        import torch
        from transformers import AlbertTokenizerFast # or AlbertTokenizer
        tokenizer = torch.load(os.path.join(args.save_to, "tokenizer.pt"))
        tokenizer = AlbertTokenizerFast(tokenizer_object=tokenizer)
    """
    global UNK_WORD, PAD_WORD, special_tokens
    PAD_WORD = "<pad>"
    UNK_WORD = '<unk>'
    special_tokens = [CLS_WORD, SEP_WORD, UNK_WORD, PAD_WORD, MASK_WORD]

    tokenizer = Tokenizer(models.Unigram())
    # normalization : replaces `` and '' by ", and lower-casing
    sequence = [normalizers.Replace("``", '"'), normalizers.Replace("''", '"')]
    if args.lowercase :
        sequence.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(sequence)
    # Metaspace pre-tokenizer : it replaces all spaces by a special character (defaulting to ▁) and then splits on that character
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    trainer = trainers.UnigramTrainer(
        vocab_size=args.vocab_size, 
        min_frequency=args.min_frequency, 
        show_progress=args.show_progress, 
        special_tokens=special_tokens
    )
    if args.path or args.split :
        tokenizer = train_from_iterator(tokenizer, args, trainer=trainer)
    else :
        tokenizer.train(files = args.files, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id(CLS_WORD)),
            ("[SEP]", tokenizer.token_to_id(SEP_WORD)),
        ],
    )
    tokenizer.decoder = decoders.Metaspace()
    tokenizer.model.save(args.save_to)
    torch.save(tokenizer, os.path.join(args.save_to, "tokenizer.pt"))
    
def train_tokenizer(args) :
    if args.from_existing != "" :
        # Using an existing tokenizer
        args.model_name = args.from_existing
        train_from_existing(args) 
    # Building a tokenizer from scratch
    elif args.type in ["byte_level_bpe"] :
        # Customize training
        general(args)
    elif args.type == "bert_word_piece" :
        # Bert Word Piece
        bert_word_piece(args)
    elif args.type == "gpt2_bpe":
        # GPT-2 BPE 
        gpt2_bpe(args)
    elif args.type == "albert_unigram_model": 
        # Albert Unigram
        albert_unigram_model(args)
        
def load_tokenizer(model_name=None, tokenizer_folder="", t_class = None, t_type = None, task = None, MAX_LEN = 512, 
                    vocab_file = None, merges_file = None) :
    """
    Load a tokenizer :
        - model_name (str) : gpt2, bert-base-uncased, ...
            * tokenier = load_tokenizer(model_name="gpt2")

        - tokenizer_folder (str) : 
            * tokenier = load_tokenizer(tokenizer_folder="/content/tokenizer")
        - t_class (str) : "roberta_tokenizer", "roberta_tokenizer_fast", "bert_tokenizer", "bert_tokenizer_fast", 
                          "gpt2_tokenizer", "gpt2_tokenizer_fast", "albert_tokenizer", "albert_tokenizer_fast"
            * tokenier = load_tokenizer(tokenizer_folder="/content/tokenizer", t_class = "bert_tokenizer_fast")

        - t_type (str) : "byte_level_bpe", "bert_word_piece", "gpt2_bpe", "albert_unigram_model"
        - task (str) : mlm, clm 
        - MAX_LEN (int)                
    """
    if model_name is not None :
        # tokenier = load_tokenizer(model_name="gpt2")
        return AutoTokenizer.from_pretrained(model_name)
    if os.path.exists(os.path.join(tokenizer_folder, "tokenizer.pt")) :
        # tokenier = load_tokenizer(tokenizer_folder="/content/gpt2")
        tokenizer = torch.load(os.path.join(tokenizer_folder, "tokenizer.pt"))
        if t_class is not None :
            #assert vocab_file or "fast" in t_class
            #assert merges_file or "bpe" in t_class
            #tokenizer = TOKENIZERS_CLASS[t_class](tokenizer_object=tokenizer, vocab_file = vocab_file, merges_file = merges_file)
            tokenizer = TOKENIZERS_CLASS[t_class](tokenizer_object=tokenizer)
        return tokenizer
    if t_type is not None :    
        assert task is None or task in ["clm", "mlm"]
        tokenizer = TOKENIZERS_TYPE[t_type](
            os.path.abspath(os.path.join(tokenizer_folder,'vocab.json')),
            os.path.abspath(os.path.join(tokenizer_folder,'merges.txt'))
        )
        # Prepare the tokenizer
        if task == "mlm" :
            tokenizer._tokenizer.post_processor = BertProcessing(
                (SEP_WORD, tokenizer.token_to_id(SEP_WORD)),
                (CLS_WORD, tokenizer.token_to_id(CLS_WORD)),
            )
        tokenizer.enable_truncation(max_length=MAX_LEN)
        if t_class is not None :
            tokenizer = TOKENIZERS_CLASS[t_class](tokenizer_object=tokenizer)
        return tokenizer
    
    if t_class is not None :
        assert task is None or task in ["clm", "mlm"]
        # Create the tokenizer from a trained one
        tokenizer = TOKENIZERS_CLASS[t_class].from_pretrained(tokenizer_folder, max_len=MAX_LEN)
        # Prepare the tokenizer
        if task == "mlm" :
            tokenizer._tokenizer.post_processor = BertProcessing(
                (SEP_WORD, tokenizer.convert_tokens_to_ids(SEP_WORD)),
                (CLS_WORD, tokenizer.convert_tokens_to_ids(CLS_WORD)),
            )
        return tokenizer
    
def build_tokenizer_from_vocab(vocab_file, t_class1, t_class2 = None) :
    """
    Build a tokenizer from vocab file :
        - vocab_file (str) :
        - t_class1 : "roberta_tokenizer", "roberta_tokenizer_fast", "bert_tokenizer", "bert_tokenizer_fast", 
                     "gpt2_tokenizer", "gpt2_tokenizer_fast", "albert_tokenizer", "albert_tokenizer_fast"
        t_class2 : ...
    """
    global UNK_WORD, PAD_WORD
    if "albert" in t_class1 :
        PAD_WORD = "<pad>"
        UNK_WORD = '<unk>'

    assert os.path.isfile(vocab_file)
    tokenizer = TOKENIZERS_CLASS[t_class1](
            vocab_file, 
            do_lower_case=True, 
            do_basic_tokenize=True, 
            never_split=None, 
            unk_token=UNK_WORD, 
            sep_token=SEP_WORD, 
            pad_token=PAD_WORD, 
            cls_token=CLS_WORD, 
            mask_token=MASK_WORD, 
            tokenize_chinese_chars=True, 
            #strip_accents=None
        )
    if t_class2 is not None :
        tokenizer = TOKENIZERS_CLASS[t_class2](tokenizer_object=tokenizer)
    return tokenizer
    
if __name__ == '__main__':

    """
    - Using an existing tokenizer
    python tokenizing.py --from_existing gpt2 --paths path_to_file1,path_to_file2 --vocab_size 25000 --save_to path_to_folder
    python tokenizing.py --from_existing gpt2 --paths wikitext --dataset_name wikitext-2-raw-v1
    - Customize training
    python tokenizing.py --type byte_level_bpe --paths path_to_file1,path_to_file2 --vocab_size 25000 --min_frequency 2 --save_to path_to_folder
    - Bert Word Piece
    python tokenizing.py --type bert_word_piece --paths path_to_file1,path_to_file2 --vocab_size 25000 --min_frequency 2 --lowercase True --save_to path_to_folder
    - GPT-2 BPE 
    python tokenizing.py --type gpt2_bpe ...
     - Albert Unigram
    python tokenizing.py --type albert_unigram_model ...
    """
    
    # parse parameters
    parser = argparse.ArgumentParser(description="Tokenizer")
    parser.add_argument('-fe', '--from_existing', type=str, default="", help="gpt2, bert-base-uncased, ...")
    parser.add_argument('-t', '--type', type=str, default="byte_level_bpe", help="")
    parser.add_argument('-lc', '--lowercase', type=bool_flag, default=True, help="") 
    parser.add_argument('-p', '--paths', type=str, help="path_to_file1,path_to_file2,... or wikitext, oscar ...\
                                                            >>> from datasets import list_datasets \
                                                            >>> datasets_list = list_datasets() \
                                                        ") 
    parser.add_argument('-dn', '--dataset_name', type=str, default="tmp", help="\
                        For wikitext : wikitext-103-v1, wikitext-2-v1, wikitext-103-raw-v1, wikitext-2-raw-v1 \
                        For oscar : unshuffled_deduplicated_af, unshuffled_deduplicated_als ... \
                        ... \
                        ") 
    parser.add_argument('-vs', '--vocab_size', type=int, help="") 
    parser.add_argument('-mf', '--min_frequency', type=int, default=1, help="")
    parser.add_argument('-bs', '--batch_size', type=int, default=1000, help="") 
    parser.add_argument('-st', '--save_to', type=dir_path, help="path_to_folder") 
    parser.add_argument('-tc', '--text_column', type=str, default="text", help="If csv, specify the text column")
    parser.add_argument('-cp', '--show_progress', type=bool_flag, default=True, help="Show progress during tokenizer training") 
    parser.add_argument('-s', '--split', type=str, default="train", help="train, validation, test")

    # generate parser / parse parameters
    args = parser.parse_args()
    args.paths = args.paths.split(',')
    args.path = None
    if len(args.paths) == 1 and not os.path.isfile(args.paths[0]) :
        args.path = args.paths[0]
    else :
        args.paths = [file_path(p) for p in args.paths]
        f1 = [f for f in args.paths if os.path.splitext(f)[1] == ".csv" ]
        f1 = csv2txt(f1, args.text_column, os.path.join(args.save_to, "files.txt"))
        f2 = [f for f in args.paths if os.path.splitext(f)[1] != ".csv" ]
        args.files = f1 + f2
    
    args.split = None if not args.split else args.split
    os.makedirs(args.save_to, exist_ok=True)

    print(args)
    train_tokenizer(args)