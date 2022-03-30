import argparse
import os
import pandas as pd
import tqdm
import ntpath

#BOS_WORD = '[BOS]'
#EOS_WORD = "[EOS]"
MASK_WORD = "[MASK]"
#MASK_WORD = "<mask>"
SEP_WORD = "[SEP]"
#SEP_WORD = "</s>"
CLS_WORD = "[CLS]"
#CLS_WORD = "<s>"
PAD_WORD = "[PAD]"
#PAD_WORD = "<pad>"
UNK_WORD = '[UNK]'
#UNK_WORD = '<unk>'
special_tokens=[CLS_WORD, PAD_WORD, SEP_WORD, UNK_WORD, MASK_WORD]

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def set_device(device) :  
    import torch  
    return torch.device(device) if device in ["cpu", "cuda"] else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")
    
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
    
def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{path} is not a valid path")
        
def str2dic_all(s) :
    """`a=int(0),b=str(y),c=float(0.1)` to {a:0, b:y, c:0.1}"""
    all_class = {"int" : int, "str" : str, "float" : float, 'bool' : bool_flag}
    s = to_none(s)
    if s is None :
        return s
    if s:
        params = {}
        for x in s.split(","):
            split = x.split('=')
            assert len(split) == 2
            val = split[1].split("(")
            _class = val[0]
            val = split[1][len(_class)+1:][:-1]
            params[split[0]] = all_class[_class](val)
    else:
        params = {}
    
    return AttrDict(params)

def str2dic(s, _class = str):
    """`a=x,b=y` to {a:x, b:y}"""
    s = to_none(s)
    if s is None :
        return s
    if s:
        params = {}
        for x in s.split(","):
            split = x.split('=')
            assert len(split) == 2
            params[split[0]] = _class(split[1])
    else:
        params = {}

    return AttrDict(params)

def str2dic_int(s):
    return str2dic(s, int)

def to_none(a):
    return None if not a or a == "_None_" else a

def intorstr(s):
    try : return int(s)
    except ValueError : return s

def path_leaf(path : str):
    """
    Returns the name of a file given its path
    https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_extension(file_path):
    return os.path.splitext(path_leaf(file_path))[-1] 

def get_path(file_item : str, extension : str, save_to : str = None):
    """
    """
    file_name = path_leaf(path = file_item)
    file_path = file_item.split(file_name)[0]
    file_name, ext = os.path.splitext(file_name) 
    if ext.replace(".", "") == extension :
        file_name = file_name + "_" + ext.replace(".", "")
    dest_file = "%s.%s"%(file_name, extension) 
    if os.path.isfile(dest_file):
        i = 1
        while os.path.isfile("%s.%s.%s"%(file_name,str(i),extension)):
            i += 1
        dest_file = "%s.%s.%s"%(file_name,str(i),extension)
    if save_to is not None :
        return os.path.join(save_to, dest_file)
    else :
        return os.path.join(file_path, dest_file)
    
def csv2txt(file_list, text_column, txt_file : None):
    """
    Convert a list of csv files to one txt file
    """
    if txt_file is None :
        txt_file = [get_path(f, "txt") for f in file_list]
        result = txt_file
    else :
        if type(txt_file) == str :
            result = [txt_file]
            txt_file = result * len(file_list)
        elif type(txt_file) == list :
            assert len(txt_file) == len(file_list)
            result = [txt_file]
    
    for file_item, tfile in zip(file_list, txt_file):
        with open(tfile, 'a') as f:
            try :
                from pandas.io.parsers import ParserError
                try :
                    df = pd.read_csv(file_item)
                except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
                    df = pd.read_csv(file_item, lineterminator='\n')
            except ImportError : # cannot import name 'ParserError' from 'pandas.io.parsers' 
                # pandas > 2.1
                df = pd.read_csv(file_item)

            for row in tqdm.tqdm(list(df.iterrows()), desc="%s" % file_item):
                text = row[1][text_column].strip()
                f.write("%s\n" % text)
                
    return result

def buid_dict_file(file_list, save_to, text_column):
    """ Build dict by space spliting"""
    import nltk
    nltk.download('punkt')
    word_to_id = {}
    for file_item in file_list:
        _, ext = os.path.splitext(path_leaf(path = file_item))
        if ext == ".csv" :
            iter = pd.read_csv(file_item).iterrows()
            iter = [row[1][text_column] for row in iter]
        else :
            iter = open(file_item, "r").readlines()
        for text in tqdm.tqdm(list(iter), desc="%s" % file_item):
            text = text.strip()
            word_list = nltk.word_tokenize(text)
            for word in word_list:
                word = word.lower()
                word_to_id[word] = word_to_id.get(word, 0) + 1
    print("Get word_dict success: %d words" % len(word_to_id))
    # write word_to_id to file
    word_dict_list = sorted(word_to_id.items(), key=lambda d: d[1], reverse=True)
    dict_file = os.path.join(save_to, "word_to_id.txt")
    with open(dict_file, 'w') as f:
        for w in special_tokens :
            f.write("%s\n"%w)
        for ii in word_dict_list:
            #f.write("%s\t%d\n" % (str(ii[0]), ii[1]))
            f.write("%s\n" % str(ii[0]))
    print("build dict finished!") 
    return dict_file


if __name__ == '__main__':
    """
    python utils.py -p path_to_file1,path_to_file2 -st path_to_folder -tc text
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Build dict file")
    parser.add_argument('-p', '--paths', type=str, help="path_to_file1,path_to_file2,... or wikitext, oscar ...") 
    parser.add_argument('-st', '--save_to', type=dir_path, help="path_to_folder") 
    parser.add_argument('-tc', '--text_column', type=str, default="text", help="If csv, specify the text column")

    # generate parser / parse parameters
    args = parser.parse_args()

    # ...
    file_list = [file_path(p) for p in args.paths.split(',')]
    os.makedirs(args.save_to, exist_ok=True)
    buid_dict_file(file_list, args.save_to, args.text_column)