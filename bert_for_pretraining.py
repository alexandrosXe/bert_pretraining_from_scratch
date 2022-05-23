import pandas as pd
import numpy as np 
import tensorflow, os, datetime
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime

from io import TextIOBase
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, BertTokenizerFast
from transformers import TFBertForMaskedLM
from tensorflow.keras.layers import Dense, TimeDistributed, Dropout
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from transformers import BertConfig
from transformers import TFBertForMaskedLM
import numpy as np
import pandas as pd
import os
import shutil
from tensorflow.keras.regularizers import l2
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import get_linear_schedule_with_warmup
from transformers import WarmUp
from transformers import create_optimizer, AdamWeightDecay




class BERT():
  def __init__(self,
                 model_path = None,
                 checkpoint_filepath = None,
                 tokenizer_path = None,
                 trainable_layers=3,
                 max_seq_length=128,
                 show_summary=False,
                 patience=3,
                 epochs=1000,
                 batch_size=32,
                 lr=2e-05,
                 session=None,
                 dense_activation = None,
                 loss='categorical_crossentropy',
                 monitor_loss = 'val_loss',
                 monitor_mode = 'min',
                 ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.session = session
        self.checkpoint_filepath = checkpoint_filepath
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True,  max_length=max_seq_length,pad_to_max_length=True)
        self.lr = lr
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.show_summary = show_summary
        self.patience=patience
        self.epochs = epochs
        self.loss = loss
        self.monitor_loss = monitor_loss
        self.monitor_mode = monitor_mode
        self.dense_activation = dense_activation
        self.earlystop = tf.keras.callbacks.EarlyStopping(monitor=self.monitor_loss,
                                                            patience=self.patience,
                                                            verbose=1,
                                                            restore_best_weights=True,
                                                            mode=self.monitor_mode)
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                            filepath=self.checkpoint_filepath,
                                                            save_weights_only=True,
                                                            monitor='val_loss',
                                                            mode='min',
                                                            save_best_only=True,
                                                            save_freq='epoch')
        self.logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.BERT = TFBertModel.from_pretrained(model_path) #, config=self.bert_config)
        self.data_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm_probability=0.15, return_tensors="np")
        #self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15, return_tensors="np")

  
  def to_bert_input(self, texts):
    input_ids, input_masks, input_segments, special_tokens_mask = [],[],[],[]
    if len(texts) > 1:
      for text in tqdm(texts):
        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_seq_length, pad_to_max_length=True, 
                                                      return_attention_mask=True, return_token_type_ids=True, return_special_tokens_mask = True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])
        special_tokens_mask.append(inputs['special_tokens_mask'])
    else:
        inputs = self.tokenizer.encode_plus(texts[0], add_special_tokens=True, max_length=self.max_seq_length, pad_to_max_length=True, 
                                                      return_attention_mask=True, return_token_type_ids=True,return_special_tokens = True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])
        special_tokens_mask.append(inputs['special_tokens_mask'])

    labels = np.asarray([x for x in input_ids])
    labels_copy = labels.copy()

    #tf_labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    masked_input_ids, masked_labels = self.data_collator.numpy_mask_tokens(labels_copy, labels_copy)#, special_tokens_mask=np.array(special_tokens_mask))

    # masked_input_ids, masked_labels = data_collator.tf_mask_tokens(tf_labels, vocab_size = 30000, mask_token_id=4, special_tokens_mask=np.asarray(special_tokens_mask, dtype='int64')) #.numpy_mask_tokens(labels_copy)
    #masked_input_ids, masked_labels = prepare_mlm_input_and_labels(labels, tokenizer)
    return (np.asarray(masked_input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')), masked_labels, labels
        
  
  def mlm_loss(self, labels, predictions):
    per_example_loss = self.masked_sparse_categorical_crossentropy(labels, predictions)
    #loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size = None)
    return per_example_loss

  def masked_sparse_categorical_crossentropy(self, y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -100))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -100))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked,
                                                           y_pred_masked,
                                                           from_logits=True)
    return loss

  def build(self, bias=0):
        in_id = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_ids", dtype='int32')
        in_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_masks", dtype='int32')
        in_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), name="segment_ids", dtype='int32')
        bert_inputs = [in_id, in_mask, in_segment]
        bert_output = self.BERT(bert_inputs).last_hidden_state
        bert_output = tf.keras.layers.BatchNormalization()(bert_output)
        bert_output = tf.keras.layers.Dropout(0.1)(bert_output)
        pred = TimeDistributed(Dense(self.tokenizer.vocab_size, activation = None, bias_regularizer=l2(0.01)))(bert_output)
        self.model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
        self.model.compile(loss=self.mlm_loss,
                      optimizer=AdamWeightDecay(learning_rate=self.lr, weight_decay_rate=0.01), run_eagerly=True)
        if self.show_summary:
            self.model.summary()

  def fit(self, train_texts, dev_texts):
      
        train_input, train_labels, _  = self.to_bert_input(train_texts)
        dev_input, dev_labels, _ = self.to_bert_input(dev_texts)
        
        train_labels = pad_sequences(maxlen=self.max_seq_length, sequences=train_labels, padding="post", value=0.0, dtype='int32')
        dev_labels = pad_sequences(maxlen=self.max_seq_length, sequences=dev_labels, padding="post", value=0.0, dtype='int32')
        self.build()
        train_info = self.model.fit(train_input,
                       train_labels,
                       validation_data=(dev_input, dev_labels),
                       epochs=self.epochs,
                       callbacks=[self.model_checkpoint_callback, self.earlystop, self.tensorboard_callback],
                       batch_size=self.batch_size,
                       class_weight=None 
                       )
        return train_info

  def predict(self, test_texts, batch_size = 10):
        test_input, test_labels, indexes_of_masked_tokens = self.to_bert_input(test_texts)
        test_labels = pad_sequences(maxlen=self.max_seq_length, sequences=test_labels, padding="post", value=0.0, dtype='int32')
        predictions = self.model.predict(test_input, batch_size = batch_size)
        print('Stopped epoch: ', self.earlystop.stopped_epoch)
        return predictions ,test_labels, indexes_of_masked_tokens
      
  #compute the error rate on predicting masked (sub)tokens
  def get_error_rate(self, preds, test_labels, masked_labels):
    error = 0
    counter = 0
    for i in range(len(preds)):
      if len(np.where(test_labels[i] != -100)[0].tolist()) == 0: #if not masked token found on the ith example
        continue
      else: 
        for index in np.where(test_labels[i] != -100)[0].tolist():
          if preds[i][index] != test_labels[i][index]:
            error+=1
          counter+=1 
    print("Total errors: ", error," out of ", counter, " masked tokens, that is :", error/counter*100, "error rate")
  
  def save_pretrained(self, path_to_save):
    #save pretrained bert
    if os.path.exists('./'+path_to_save):
    	shutil.rmtree('./'+path_to_save)
    os.mkdir('./'+path_to_save)
    self.BERT.save_pretrained(path_to_save)

  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)

    #save entire model
  def save_model(self, path):
    self.model.save(path)
  #load entire model
  def load_model(self, path):
    self.build()
    self.model.load_model(path)
