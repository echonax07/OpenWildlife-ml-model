from transformers import BertConfig, BertModel
from transformers import AutoTokenizer

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# config.save_pretrained("your path/bert-base-uncased")
# model.save_pretrained("your path/bert-base-uncased")
# tokenizer.save_pretrained("your path/bert-base-uncased")

import nltk
nltk.download('punkt', download_dir='/home/m32patel/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='/home/m32patel/nltk_data')