## 1. Setting
```bash
git clone https://github.com/Tikquuss/lm_hf
cd lm_hf

python3 -m install pip
pip install -r requirements.txt
```

## 2. Build a tokenizer from scratch if you are not going to use a pre-trained model (supports txt and csv)  

See [tokenizing.py](src/tokenizing.py) for all other parameters (and descriptions).
```bash
st=my/save/path
mkdir -p $st

datapath=/path/to/data
text_column=text
python -m src.tokenizing -fe gpt2 -p ${datapath}/data_train.csv,${datapath}/data_val.csv,${datapath}/data_test.csv -vs 25000 -mf 2 -st $st -tc $text_column

#python -m src.tokenizing -fe bert-base-uncased -p wikitext -dn wikitext-2-raw-v1 --vocab_size 25000 -st $st

# ...
```

The tokenizer will be saved in ```${save_to}/tokenizer.pt```.

## 3. Dictionary (work, but deprecated for the moment)

You can, instead of pre-training a tokenizer, build a simple vocabulary (by dividing the sentences according to the whitespace character - 
default option, or by dividing sentences into phonemes, ...), then build the tokenizer with this vocabulary during the training/evaluation (```tokenizer_params="vocab_file=str(${save_to}/word_to_id.txt),t_class=str(bert_tokenizer),..."```).

```bash
st=my/save/path
mkdir -p $st

datapath=/path/to/data
text_column=text

python -m src.utils -p ${datapath}/data_train.csv,${datapath}/data_val.csv,${datapath}/data_test.csv -st $st -tc $text_column
```

But this option is not recommended for the moment (any deep sanitary check has been done so far).

## 4. Train and/or evaluate a model (from scratch or from a pre-trained model and/or tokenizer)  
See [trainer.py](src/trainer.py) and [train.sh](train.sh) for all other parameters (and descriptions)
```bash
. train.sh
```

## 5. TensorBoard (visualize the evolution of the loss/acc/... per step/epoch/...)
See https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
```
%load_ext tensorboard

%tensorboard --logdir ${log_dir}/${task}/lightning_logs
```

## 6. Prediction

To generate texts or fill the masks, you have to use the ```predict_params``` parameter.
By default, this will be done on the test dataset (or the dataset specified with the ```split``` parameter), but it is better to put your examples in a text file, csv, json (...) and use it instead of the test dataset (```test_data_files``` parameter).
Don't forget to set the ```group_texts``` parameter to ```False``` in this case, and make sure that the length of the prompts or sentences (and the value of the ```max_length``` parameter) does not exceed the value of the ```max_position_embeddings```/```n_positions```/ ... parameter of your model.

- For example for text generation, the file can be in the following form:
    * for a text file :
    ```
    prompt 1
    prompt 2
    ...
    ```
    * for a csv file :
    ```
    text_column | ...
    prompt 1    | ...
    prompt 2    | ...
    ...	    | ...
    ```

- For the mask filling, replace the prompts below by the sentences on which to do the MLM


The result will be stored by default in the ```${log_dir}/${task}/predict.txt``` file, but you can change this path by adding this value to the ```predict_params``` parameter:
```bash
predict_params="...,output_file=str(my_path/file.txt),..."
```
