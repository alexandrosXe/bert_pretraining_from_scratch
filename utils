from tqdm.auto import tqdm
from pathlib import Path
import os
from tokenizers import BertWordPieceTokenizer
import shutil


def save_corpus_to_files(texts, path_to_save):
  text_data = []
  file_count = 0
  for i, text in tqdm(enumerate(texts)):
    text_data.append(text)
    if len(text_data) == 10_000:
        # once we git the 10K mark, save to file
        with open(f'{path_to_save}/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1
  #write the leftovers
  with open(f'{path_to_save}/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(text_data))
    
    
#Get a list of paths to each file in our oscar_it directory.
# Now we move onto training the tokenizer. 
# We use a byte-level Byte-pair encoding (BPE) tokenizer. 
# This allows us to build the vocabulary from an alphabet of single bytes, 
# meaning all words will be decomposable into tokens.

def train_tokenizer(paths_of_data, path_to_save_tokenizer):
  # initialize
  tokenizer = BertWordPieceTokenizer(
      clean_text=True,
      handle_chinese_chars=False,
      strip_accents=False,
      lowercase=True
  )
  # and train
  tokenizer.train(files=paths_of_data, vocab_size=30000, min_frequency=2,
                  limit_alphabet=1000, wordpieces_prefix='##',
                  special_tokens=[
                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
  if os.path.exists('./'+path_to_save_tokenizer):
    	shutil.rmtree('./'+path_to_save_tokenizer)
  os.mkdir('./'+path_to_save_tokenizer)
  tokenizer.save_model(path_to_save_tokenizer) 
