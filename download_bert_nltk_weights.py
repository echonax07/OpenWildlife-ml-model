from transformers import BertConfig, BertModel
from transformers import AutoTokenizer
import os 

os.makedirs("checkpoints", exist_ok=True)
config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config.save_pretrained("checkpoints/bert/bert-base-uncased")
model.save_pretrained("checkpoints/bert/bert-base-uncased")
tokenizer.save_pretrained("checkpoints/bert/bert-base-uncased")

import nltk
# Download NLTK data
nltk.download('punkt',download_dir='checkpoints/nltk_data')
nltk.download('averaged_perceptron_tagger',download_dir='checkpoints/nltk_data')
nltk.download('averaged_perceptron_tagger_eng',download_dir='checkpoints/nltk_data')

# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# nltk.download('punkt', download_dir='/home/m32patel/nltk_data')
# nltk.download('averaged_perceptron_tagger', download_dir='/home/m32patel/nltk_data')

# user_home = os.path.expanduser("~")



# nltk.download('punkt', download_dir=f'{user_home}/nltk_data')
# nltk.download('averaged_perceptron_tagger', download_dir=f'{user_home}/nltk_data')
# nltk.download('averaged_perceptron_tagger_eng', download_dir=f'{user_home}/nltk_data')

# nltk.download('punkt_tab', download_dir=f'{user_home}/nltk_data')


# from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
# import os
# import nltk

# # Download and save CLIP model components
# model_name = "openai/clip-vit-base-patch32"  # Choose your CLIP variant

# Create save directory
# save_path = "checkpoints/clip/clip-vit-base-patch32"
# os.makedirs(save_path, exist_ok=True)

# Load and save components
# config = CLIPTextConfig.from_pretrained(model_name)
# model = CLIPTextModel.from_pretrained(model_name, config=config)
# tokenizer = CLIPTokenizer.from_pretrained(model_name)

# config.save_pretrained(save_path)
# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)

# Download NLTK data (same as before)
# user_home = os.path.expanduser("~")
# nltk.download('punkt', download_dir=f'{user_home}/nltk_data')
# nltk.download('averaged_perceptron_tagger', download_dir=f'{user_home}/nltk_data')



# # Download and save OPENCLIP model components
# # model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"  # Choose your CLIP variant
# model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

# # Create save directory
# save_path = "checkpoints/laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
# os.makedirs(save_path, exist_ok=True)

# # Load and save components
# config = CLIPTextConfig.from_pretrained(model_name)
# model = CLIPTextModel.from_pretrained(model_name, config=config)
# tokenizer = CLIPTokenizer.from_pretrained(model_name)

# config.save_pretrained(save_path)
# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)

# # Download NLTK data (same as before)
# user_home = os.path.expanduser("~")
# nltk.download('punkt', download_dir=f'{user_home}/nltk_data')
# nltk.download('averaged_perceptron_tagger', download_dir=f'{user_home}/nltk_data')