from transformers import BertConfig, BertModel
from transformers import AutoTokenizer
import os 
config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config.save_pretrained("checkpoints/bert/bert-base-uncased")
model.save_pretrained("checkpoints/bert/bert-base-uncased")
tokenizer.save_pretrained("checkpoints/bert/bert-base-uncased")

import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# nltk.download('punkt', download_dir='/home/m32patel/nltk_data')
# nltk.download('averaged_perceptron_tagger', download_dir='/home/m32patel/nltk_data')

user_home = os.path.expanduser("~")



# nltk.download('punkt', download_dir=f'{user_home}/nltk_data')
# nltk.download('averaged_perceptron_tagger', download_dir=f'{user_home}/nltk_data')
nltk.download('averaged_perceptron_tagger_eng', download_dir=f'{user_home}/nltk_data')

nltk.download('punkt_tab', download_dir=f'{user_home}/nltk_data')

