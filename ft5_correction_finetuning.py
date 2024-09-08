#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().system(' python --version')


# # init
# NOTE: 
# - This was tested with python version 3.12.2 on Mac Sonoma 14.1. Needed to import a downgraded version of numpy to support downgraded version of wandb.
# - This was tested with python version 3.10.11 on Windows 11.

# In[ ]:


#get_ipython().system(' pip install transformers==4.36.2')
#get_ipython().system(' pip install wandb==0.17.1')
#get_ipython().system(' pip install accelerate==0.31.0')
#get_ipython().system(' pip install rouge_score==0.1.2')
#get_ipython().system(' pip install sacrebleu==2.4.2')
#get_ipython().system(' pip install sentencepiece==0.1.99')
#get_ipython().system(' pip install evaluate==0.4.2')
#get_ipython().system(' pip install nltk==3.8.1')
#get_ipython().system(' pip install jiwer==3.0.4')
#get_ipython().system(' pip install numpy==1.26.4')


# In[ ]:


import random
import numpy as np
import pandas as pd
import torch
import correction_palette as palette
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments
from correction_palette import *
from cursor_t5_modeling import CursorT5ForConditionalGeneration
import evaluate
import wandb
import os
import copy
import gc
os.environ["WANDB_NOTEBOOK_NAME"] = "t5_ft5_correction_sweeps.ipynb"
# wandb login
wandb.login()
# Default 'online', use 'offline' to avoid network issues.
wandb_mode = 'online'


# In[ ]:


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    gpu_id = 0

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device(gpu_id)
    print(f"Using CUDA with {torch.cuda.get_device_name(i)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


# # Configure stuffs

# In[ ]:


save_dir = './'  # save directory for pretty much anything that is saved by this program (models, tokenizers, logs, etc.)
models_dir = f'{save_dir}saved_models/'
data_dir = f'{save_dir}DELETION-INSERTION-MULTIPLE-REPLACEMENT-SINGLE/'

base_model_name = 'google/flan-t5-base'

NUM_SAVES = 2

PORTION = 1 # proportion ofdatasets to use (note: applies to each split)

random_seed = 42
palette.MAX_SOURCE_TOKENS = 512
palette.MAX_NEW_TOKENS = 512
palette.CURSOR_TOKEN = '<|>'
palette.TEXT_SEP_TOKEN = '<||>'
custom_special_tokens_dict = {'additional_special_tokens': [palette.CURSOR_TOKEN, palette.TEXT_SEP_TOKEN]}

gen_kwargs_override = {'num_beams': 24, 'max_new_tokens': palette.MAX_NEW_TOKENS, 'early_stopping': True, 'num_return_sequences': 24}  # these values might be changed in 'Sweep Configuration'


# # Computing Metrics

# In[ ]:


import difflib
import re


def prefix_key(k: str, prefix: str, sep: str) -> str:
    return f'{prefix}{sep}{k}'

def get_unique_words(text):
    #words = set(text.replace('_', ' ').replace(',', ' ').replace(':', ' ').replace(';', ' ').replace('.', ' ').replace('?', ' ').replace('(', ' ').split().split())
    list_words = re.split(',|\ |_|-|!|\+|\.|\\|\*|\?|\[|\^|\]|\$|\(|\)|\{|\}|\=|\||\:', text)
    words = set(list_words)
    if '' in words:
        words.remove('')
    return words

def get_split_index(split_list,index):
    start_index = 0
    end_index = 0
    for i in range(0,index+1):
        if i == 0:
            start_index = 0
            end_index = len(split_list[i])
        else:
            start_index+=len(split_list[i-1])+1
            end_index= start_index+len(split_list[i])
    return start_index,end_index

def get_diff_start_end(original,new):

    original = ' '+original
    new = ' ' +new
    s1 = original.split(' ')
    s2 = new.split(' ')
    matcher = difflib.SequenceMatcher(a=s1, b=s2)
    #print("Matching Sequences:")
    '''
    a_last_end = 0
    count = 0
    for match in matcher.get_matching_blocks():
        #print("Match             : {}".format(match))
        #print("Matching Sequence : {}".format(s1[match.a:match.a+match.size]))
        #print(match.a,match.a+match.size)
        if count>0:
            if match.a-a_last_end>0:
                #print("missmatch",a_last_end,match.a)
                #print("Miss_matching Sequence : {}".format(s1[a_last_end:match.a]))
                start,e = get_split_index(s1,a_last_end)
                s,end = get_split_index(s1,match.a-1)
                print("mismatch start end:",start,end)
                print("***miss match segment:", string1[start:end])


        count+=1
        a_last_end=match.a+match.size
    '''
    b_last_end = 0
    count = 0
    start_list = []
    end_list = []
    for match in matcher.get_matching_blocks():
        #print("Match             : {}".format(match))
        #print("Matching Sequence : {}".format(s2[match.b:match.b+match.size]))
        #print(match.b,match.b+match.size)
        if count>0:
            if match.b-b_last_end>0:
                #print("missmatch",b_last_end,match.b)
                #print("Miss_matching Sequence : {}".format(s2[b_last_end:match.b]))
                start,_ = get_split_index(s2,b_last_end)
                _,end = get_split_index(s2,match.b-1)
                start_list.append(start-1)
                end_list.append(end-1)

                #print("mismatch start end:",start,end)
                #print("***miss match segment:", new[start:end])
        count+=1
        b_last_end=match.b+match.size

    return start_list, end_list

def filter_corrections(response, sentence_without_marker,data):
    filtered_corrections = []
    filtered_orig_idx = []
    diff_start_list = []
    diff_end_list = []
    filter_idx = 0
    for item in response['corrections']:
        if item.strip(" ") != item:
            item = item.strip(" ")
        if (item!=sentence_without_marker):

            if (item not in filtered_corrections):
                item_words = get_unique_words(item.lower());
                original_and_input_words = get_unique_words(sentence_without_marker.lower()+" "+data['part'].lower())
                #print("item_words",item_words)
                #print("original_and_input_words",original_and_input_words)
                if item_words.issubset(original_and_input_words):
                    if (data['part'].lower() in item.lower()):
                        # and (response['correction probabilities'][filter_idx]>0):
                        if (len((data['part']+" "+sentence_without_marker).split()) >= len(item.split())):
                            filtered_corrections.append(item)
                            filtered_orig_idx.append(filter_idx)

                            (start_list, end_list) = get_diff_start_end(sentence_without_marker,item)
                            diff_start_list.append(start_list)
                            diff_end_list.append(end_list)
                            '''
                            if (response['correction probabilities'][filter_idx]>0.7):
                                break
                            else:
                                print("prob <0.7")
                                print(response['correction probabilities'][filter_idx], item)
                            '''


        filter_idx += 1
    return filtered_corrections, filtered_orig_idx, diff_start_list, diff_end_list

def F1_score(ground_truth, result_list):
    num_of_results = float(len(result_list))
    recall = 0
    precision = 0
    if ground_truth in result_list:
        recall = 1
        precision = 1.0/num_of_results;
    else:
        recall = 0
        precision = 0

    F1 = 2*recall*precision/(recall+precision+0.00000000001)
    return F1,recall,precision

f1,recall,precision = F1_score('how are you',[])
print('recall',recall)
print("precision",precision)
print("F1",f1)


string1 ="It is not warm. It is summer."
string2 ="It is very warm. It is winter."
get_diff_start_end(string1,string2)


# In[ ]:


def filter_invalid_samples(_label_str, _input_strs, _beam_str, _best_beam_str):
    label_str = []
    input_strs = []
    best_beam_str = []
    beam_str = []
    for i in range(len(_label_str)):
        e_sentence, phrase = _input_strs[i]
        e_sentence, phrase = tokenizer.decode(tokenizer(e_sentence)['input_ids'], skip_special_tokens=True), tokenizer.decode(tokenizer(phrase)['input_ids'], skip_special_tokens=True)
        label = _label_str[i]
        if label == e_sentence:
            continue

        beam_group = _beam_str[i]
        label = label

        d_response = {'corrections': beam_group}
        d_data = {'part': phrase}

        pred_slice_before = beam_group[:5]
        num_correct_before = label in pred_slice_before
        filter_output = filter_corrections(d_response, e_sentence, d_data)

        pred_slice_after = filter_output[0][:5]
        num_correct_after = label in pred_slice_after
        if num_correct_after < num_correct_before:
            print(f'skipped (sent: {e_sentence}, phra: {phrase})')
            continue

        label_str.append(label)
        input_strs.append((e_sentence, phrase))
        best_beam_str.append(_best_beam_str[i])
        beam_str.append(_beam_str[i])
    return label_str, input_strs, best_beam_str, beam_str

# filter beams based on criterion. save filtered_ids so that we can simulate slicing the top n beams and then filtering without recomputing/refiltering.
def filter_beams(label_str, input_strs, beam_str, best_beam_str):
    filtered_ids = []

    #print(f'beam_str: {np.array(beam_str).shape}')
    #print(f'label_str: {np.array(label_str).shape}')
    #print(f'input_strs: {np.array(input_strs).shape}')
    #print(f'input_strs: {input_strs}')
    for i in range(len(beam_str)): # filter invalid beams
        beam_group = beam_str[i]
        label = label_str[i]
        sentence, phrase = input_strs[i]

        d_response = {'corrections': beam_group}
        d_data = {'part': phrase}

        pred_slice_before = beam_group[:5]
        num_correct_before = label in pred_slice_before
        filter_output = filter_corrections(d_response, sentence, d_data)
        beam_str[i] = filter_output[0]
        if len(beam_str[i]) > 0:
            best_beam_str[i] = beam_str[i][0]
        filtered_ids.append(filter_output[1])
        pred_slice_after = beam_str[i][:5]
        num_correct_after = label in pred_slice_after
        if num_correct_after < num_correct_before:
            print(f'impossible')
    return filtered_ids, beam_str, best_beam_str

# computing f1, precision, and recall
def compute_fpr(_beam_str, label_str, filtered_ids, prefix: str = 'n_samp'):
    beam_str = copy.deepcopy(_beam_str)
    n_metrics = dict()
    n_table = []
    for j in range(24, 0, -1): # iterate over possible num_samples (to obtain n_metrics)
        f1_sum = 0
        recall_sum = 0
        precision_sum = 0

        for i in range(len(beam_str)):
            if filtered_ids is not None:
                while len(beam_str[i]) > 0 and filtered_ids[i][len(beam_str[i]) - 1] >= j:
                    # remove from end based on filtered_ids because filtered_ids represents the original indices of the beams.
                    # this way, we don't have to keep re-slicing the top n beams and re-filtering
                    beam_str[i].pop()

            metric_output = F1_score(label_str[i], beam_str[i])
            f1_sum += metric_output[0]
            recall_sum += metric_output[1]
            precision_sum += metric_output[2]
            if filtered_ids is None:
                beam_str[i].pop()

        f1 = f1_sum / len(beam_str)
        recall = recall_sum / len(beam_str)
        precision = precision_sum / len(beam_str)

        n_table.append({'n': j, 'f1': f1, 'recall': recall, 'precision': precision})

        n_metrics[f'n{j}/f1'] = f1
        n_metrics[f'n{j}/recall'] = recall
        n_metrics[f'n{j}/precision'] = precision

    n_metrics = {prefix_key(k, prefix,'/'): v for k, v in n_metrics.items()}

    # display tables
    n_table = pd.DataFrame(n_table).iloc[::-1].reset_index(drop=True)
    return n_metrics, n_table


# In[ ]:


from IPython.display import display, HTML

bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
cer = evaluate.load('cer')
tkacc = TopKAccuracy()
tokenizer = T5Tokenizer.from_pretrained(base_model_name)

def compute_metrics(gen_pred: GenPrediction):
    label_ids = gen_pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_ids = gen_pred.predictions
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id
    _input_strs = gen_pred.inputs_strs
    _label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    for lstr in _label_str:
        lstr = ' '.join([word for word in lstr.split(' ') if word != '|>' and word != '|>.'])

    _pred_str = tokenizer.batch_decode(np.array(pred_ids).reshape(-1, len(pred_ids[0][0])), skip_special_tokens=True)
    _pred_str = np.array(_pred_str, dtype=str)


    _best_beam_str = _pred_str[::gen_kwargs_override['num_return_sequences']].tolist()
    _beam_str = _pred_str.reshape(-1, gen_kwargs_override['num_return_sequences']).tolist()


    label_str, input_strs, best_beam_str, beam_str = filter_invalid_samples(_label_str, _input_strs, _beam_str, _best_beam_str)

    # compute metrics before filtering
    tkacc_metrics_unfiltered = tkacc.compute(predictions=beam_str, references=label_str, k_metrics=[1, 3, 5], strict_k=False)
    bleu_metrics_unfiltered = bleu.compute(predictions=best_beam_str, references=label_str)
    rouge_metrics_unfiltered = rouge.compute(predictions=best_beam_str, references=label_str)
    cer_metric_unfiltered = {'cer_unfiltered': cer.compute(predictions=best_beam_str, references=label_str)}
    bleu_metrics_unfiltered['ga_precision'] = bleu_metrics_unfiltered['bleu'] / bleu_metrics_unfiltered['brevity_penalty']  # global average precision (clipped). same as bleu but without brevity penalty.
    n_metrics_unfiltered, n_table_unfiltered = compute_fpr(beam_str, label_str, None, prefix='fpr_unfiltered')

    # filter bad results based on filtering criterion. also save filtered_ids to avoid unecessary recomputations.
    filtered_ids, beam_str, best_beam_str = filter_beams(label_str, input_strs, beam_str, best_beam_str)

    # compute metrics after filtering
    tkacc_metrics_filtered = tkacc.compute(predictions=beam_str, references=label_str, k_metrics=[1, 3, 5], strict_k=False)
    bleu_metrics_filtered = bleu.compute(predictions=best_beam_str, references=label_str)
    rouge_metrics_filtered = rouge.compute(predictions=best_beam_str, references=label_str)
    cer_metric_filtered = {'cer_filtered': cer.compute(predictions=best_beam_str, references=label_str)}
    bleu_metrics_filtered['ga_precision'] = bleu_metrics_filtered['bleu'] / bleu_metrics_filtered['brevity_penalty']  # global average precision (clipped). same as bleu but without brevity penalty.
    n_metrics_filtered, n_table_filtered = compute_fpr(beam_str, label_str, filtered_ids, prefix='fpr_filtered')

    # filter irrelevant metrics
    bleu_metrics_unfiltered = {k: v for k, v in bleu_metrics_unfiltered.items() if k not in ['precisions', 'brevity_penalty']}
    bleu_metrics_filtered = {k: v for k, v in bleu_metrics_filtered.items() if k not in ['precisions', 'brevity_penalty']}

    # add prefixes
    bleu_metrics_unfiltered = {prefix_key(k, 'bleu_unfiltered', '/'): v for k, v in bleu_metrics_unfiltered.items()}
    bleu_metrics_filtered = {prefix_key(k, 'bleu_filtered', '/'): v for k, v in bleu_metrics_filtered.items()}
    rouge_metrics_unfiltered = {prefix_key(k, 'rouge_unfiltered', '/'): v for k, v in rouge_metrics_unfiltered.items()}
    rouge_metrics_filtered = {prefix_key(k, 'rouge_filtered', '/'): v for k, v in rouge_metrics_filtered.items()}
    tkacc_metrics_unfiltered = {prefix_key(k, 'tkacc_unfiltered', '/'): v for k, v in tkacc_metrics_unfiltered.items()}
    tkacc_metrics_filtered = {prefix_key(k, 'tkacc_filtered', '/'): v for k, v in tkacc_metrics_filtered.items()}

    # construct a table for non-FPR metrics
    metric_table = pd.DataFrame([{
        **cer_metric_unfiltered,
        **cer_metric_filtered,

        **tkacc_metrics_unfiltered,
        **tkacc_metrics_filtered,

        **rouge_metrics_unfiltered,
        **rouge_metrics_filtered,

        **bleu_metrics_unfiltered,
        **bleu_metrics_filtered
    }])

    # display tables for FPR metrics
    display(HTML(f"<h2>FPR metrics BEFORE filtering</h2>" + n_table_unfiltered.to_html(index=False)))
    display(HTML(f"<h2>FPR metrics AFTER filtering</h2>" + n_table_filtered.to_html(index=False)))

    # display non-FPR metrics
    display(HTML(metric_table.to_html(index=False)))

    # construct final dictionary for metrics
    metric_dict = {
        **cer_metric_unfiltered,
        **cer_metric_filtered,

        **tkacc_metrics_unfiltered,
        **tkacc_metrics_filtered,

        **rouge_metrics_unfiltered,
        **rouge_metrics_filtered,

        **bleu_metrics_unfiltered,
        **bleu_metrics_filtered,

        **n_metrics_unfiltered,
        **n_metrics_filtered
    }
    return metric_dict


# # Sweep Code
# Run this if you'd like to generate the training samples from the training data "on the fly" during training. The random behavior has been minimized with set seeds.

# ## Sweep Configuration

# In[ ]:


# method
sweep_config_ablation = {
    'method': 'grid',
    'metric': {
        'goal': 'minimize',
        'name': 'eval/bloss'
        },
    'name': 'cursor mask text correction'
}

# hyperparameters
parameters_dict_ablation = {
    'epochs': {
        'value': 16
    },
    'train_batch_size': {
        'value': 4
    },
    'eval_batch_size': {
        'value': 24
    },
    'gradient_accumulation_steps':{
        'value': 4
    },
    'learning_rate': {
        'value': 5e-5
    },
    'weight_decay': {
        'value': 0.001
    },
    'optim': {
        'value': 'adafactor'
    },
    'logging_rate': {
        'value': 0.01
    },
    'eval_rate': {
        'value': 0.01
    },
    'save_rate': {
        'value': 0.01
    },
# start of model characteristics
    'param_group': {
        'values': [
            # changing correction phrase sampling strategies
            #('uniform-multiple correction', 'google/flan-t5-base',
            # {'correction_strategy': 'uniform-multiple', 'correction_distrib': (0, 1), 'cursor_strategy': 'normal', 'cursor_relax': 5, 'invert_case_prob': 0.5, 'log_stuff': False}),
            #('minimum-multiple correction', 'google/flan-t5-base',
            # {'correction_strategy': 'minimum-multiple', 'correction_distrib': (0, 1), 'cursor_strategy': 'normal', 'cursor_relax': 5, 'invert_case_prob': 0.5, 'log_stuff': False}),

            # changing cursor sampling strategies
            #('uniform cursor', 'google/flan-t5-base',
            # {'correction_strategy': 'normal-multiple', 'correction_distrib': (0, 1), 'cursor_strategy': 'uniform', 'cursor_relax': 5, 'invert_case_prob': 0.5, 'log_stuff': False}),
            #('none cursor', 'google/flan-t5-base',
            # {'correction_sztrategy': 'normal-multiple', 'correction_distrib': (0, 1), 'cursor_strategy': 'none', 'cursor_relax': 5, 'invert_case_prob': 0.5, 'log_stuff': False}),

            # removing case-inversion
            #('0 case-inversion probability', 'google/flan-t5-base',
            # {'correction_strategy': 'normal-multiple', 'correction_distrib': (0, 1), 'cursor_strategy': 'normal', 'cursor_relax': 5, 'invert_case_prob': 0, 'log_stuff': False})#,

            # changing size of base FLAN-T5 model
            #('test run', 'google/flan-t5-base',
            # {'correction_strategy': 'normal-multiple', 'correction_distrib': (0, 1), 'cursor_strategy': 'normal', 'cursor_relax': 5, 'invert_case_prob': 0.5, 'log_stuff': False}),
            #('small', 'google/flan-t5-small',
            # {'correction_strategy': 'normal-multiple', 'correction_distrib': (0, 1), 'cursor_strategy': 'normal', 'cursor_relax': 5, 'invert_case_prob': 0.5, 'log_stuff': False}),
            #('large', 'google/flan-t5-large',
            # {'correction_strategy': 'normal-multiple', 'correction_distrib': (0, 1), 'cursor_strategy': 'normal', 'cursor_relax': 5, 'invert_case_prob': 0.5, 'log_stuff': False}),
            #('x-large', 'google/flan-t5-xl',
            # {'correction_strategy': 'normal-multiple', 'correction_distrib': (0, 1), 'cursor_strategy': 'normal', 'cursor_relax': 5, 'invert_case_prob': 0.5, 'log_stuff': False})
            ('cursor_mask cross-masked-uniform', base_model_name,
             {'correction_strategy': 'normal-multiple', 
              'correction_distrib': (0, 1), 
              'cursor_strategy': 'normal',
              'cursor_bias_type': 'cross-masked-uniform',
              'cursor_relax': 5, 
              'cursor_rep': 'mask', 
              'postprocessor': 'none', 
              'invert_case_prob': 0.5, 
              'log_stuff': False}
            )
        ]
    }
# end of model characteristics
}

sweep_config_ablation['parameters'] = parameters_dict_ablation

sweep_config = sweep_config_ablation


# In[ ]:


sweep_config


# ## Train func

# In[ ]:


def train(config=None):
    with wandb.init(config=config, tags=['save run'], name="test", mode=wandb_mode):
        # set random seeds
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        np.random.seed(random_seed)
        random.seed(random_seed)

        # set sweep configuration
        config = wandb.config
        wandb.run.name = config.param_group[0]

        # run/model name
        name = config.param_group[0]

        # model path
        model_path = f"{models_dir}{sweep_config['name']}/{name}"
        #model_hub_path = f"{sweep_configp['name']/{name}}"

        # set the model size
        model_name = config.param_group[1]
        print(f'config.param_group[2]: {config.param_group[2]}')


        tokenizer = T5Tokenizer.from_pretrained(model_name)
        if config.param_group[2]['cursor_rep'] in ['token', 'mask']:
            print(f"added {tokenizer.add_special_tokens(custom_special_tokens_dict)} custom special tokens to tokenizer")

        def init_model() -> T5ForConditionalGeneration:
            model = CursorT5ForConditionalGeneration.from_pretrained(model_name, config.param_group[2]['cursor_bias_type'] )
            model.resize_token_embeddings(len(tokenizer))
            print(f'init model: base_model_name: {model_name}')
            return model

        # sampling strategies
        print(config.param_group[2]['cursor_strategy'] if config.param_group[2]['cursor_strategy'] != 'uniform' else 'normal')
        train_params = dict(config.param_group[2])
        train_params['inference'] = False
        inference_params = dict(config.param_group[2])
        inference_params['inference'] = True
        inference_params['cursor_strategy'] = 'normal-multiple'
        inference_params['correction_distrib'] = (0, 1)
        inference_params['cursor_strategy'] = 'normal' if train_params['cursor_strategy'] != 'none' else 'none'
        inference_params['cursor_relax'] = 5
        inference_params['cursor_rep'] = train_params['cursor_rep'] if train_params['cursor_rep'] is not None else 'token'
        inference_params['invert_case_prob'] = 0.5
        inference_params['log_stuff'] = False

        print(f'train_params: {train_params}')
        print(f'inference_params: {inference_params}')

        train_sampling_strategy = EditSamplingStrategy(**train_params)
        val_sampling_strategy = EditSamplingStrategy(**inference_params)
        test_sampling_strategy = EditSamplingStrategy(**inference_params)

        print(f'train_sampling_strategy: {train_sampling_strategy.postprocess_inner}')

        # datasets
        train_dataset = CorrectionDatasetWithEdits(f'{data_dir}train_data.csv',
                                                   tokenizer=tokenizer,
                                                   sampling_strategy=train_sampling_strategy,
                                                   scale=PORTION,
                                                   random_state=random_seed,
                                                   save_dataset=True,
                                                   one_draw=False)
        val_dataset = CorrectionDatasetWithEdits(f'{data_dir}val_data.csv',
                                                 tokenizer=tokenizer,
                                                 sampling_strategy=val_sampling_strategy,
                                                 scale=PORTION,
                                                 random_state=random_seed,
                                                 save_dataset=True,
                                                 one_draw=True)
        test_dataset = CorrectionDatasetWithEdits(f'{data_dir}test_data.csv',
                                                  tokenizer=tokenizer,
                                                  sampling_strategy=test_sampling_strategy,
                                                  scale=PORTION,
                                                  random_state=random_seed,
                                                  save_dataset=True,
                                                  one_draw=True)
        ex_train_sample = train_dataset[0]
        ex_train_input = ex_train_sample['input_str']
        ex_train_label = ex_train_sample['label_str']
        print(f'example train input string: {ex_train_input}')
        print(f'example train input string: {ex_train_label}')

        # set training arguments
        training_args = Seq2SeqTrainingArguments(
            report_to='wandb',

            output_dir=''.join([model_path, '/checkpoints/']),

            num_train_epochs=config.epochs,

            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,

            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,

            gradient_accumulation_steps=config.gradient_accumulation_steps,

            optim=config.optim,

            logging_strategy='steps',
            evaluation_strategy='steps',
            save_strategy='steps',

            logging_steps=config.logging_rate,
            eval_steps=config.eval_rate,
            save_steps=config.save_rate,

            load_best_model_at_end=True,
            metric_for_best_model='loss',
            greater_is_better=False,
            save_total_limit=NUM_SAVES,

            predict_with_generate=True,
            prediction_loss_only=True, # disable using .generate() for evaluation steps during training because it is very slow

            no_cuda=False
        )

        # define training loop
        trainer = EarlyStoppingSeq2SeqTrainer(
            model=init_model(),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda gen_preds : {},
            gen_kwargs_override={**gen_kwargs_override, 'decoder_start_token_id': 0},
            log_stuff=True
        )

        # initial evaluation step
        #trainer.evaluate()

        # start training loop
        trainer.train()

        # load checkpoint with lowest validation loss
        model = CursorT5ForConditionalGeneration.from_pretrained(trainer.state.best_model_checkpoint, config.param_group[2]['cursor_bias_type'])

        # check vocab
        print(f'model vocab size: {model.config.vocab_size}')
        print(f'tokenizer vocab: {len(tokenizer)}')

        # save best checkpoint
        model.save_pretrained(f'{model_path}/')

        trainer.compute_metrics = compute_metrics
        trainer.args.predict_with_generate = True
        trainer.args.include_inputs_for_metrics = True
        trainer.args.prediction_loss_only = False
        trainer.args.per_device_eval_batch_size = 4

        # test
        test_evaluation = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
        print(f"test evaluation: \n{test_evaluation}\n")

        # save test results
        test_evaluation_df = pd.DataFrame(data=[test_evaluation], columns=test_evaluation.keys())
        test_evaluation_df.to_csv(f'{model_path}/test_results.csv', index=False)

        print(f"saved best checkpoint: {trainer.state.best_model_checkpoint.split('/')[-1]}")

        train_dataset_df = pd.DataFrame(data=train_dataset.dataset_mem, columns=['input', 'label'])
        train_dataset_df.to_csv(f'{model_path}/train_dataset.csv', index=False)

        val_dataset_df = pd.DataFrame(data=val_dataset.dataset_mem, columns=['input', 'label'])
        val_dataset_df.to_csv(f'{model_path}/val_dataset.csv', index=False)

        test_dataset_df = pd.DataFrame(data=test_dataset.dataset_mem, columns=['input', 'label'])
        test_dataset_df.to_csv(f'{model_path}/test_dataset.csv', index=False)


# ## Train agent

# In[ ]:


wandb.finish()
sweep_id = wandb.sweep(sweep_config, project='vr-txt-corr')
wandb.agent(sweep_id, train)

