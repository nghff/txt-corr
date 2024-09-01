import ast
import re
import random
from collections import defaultdict
from typing import Dict, Optional, List, Union, Tuple, Any, Callable

import datasets
import numpy as np
import pandas as pd
import torch.cuda
from torch import nn, LongTensor, FloatTensor, tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Seq2SeqTrainer, T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer
from transformers.trainer_pt_utils import *
from tqdm import tqdm
from transformers.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init
from transformers.trainer_utils import *

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")
MAX_SOURCE_TOKENS: int = 512
MAX_NEW_TOKENS: int = 512
TEXT_SEP_TOKEN: str = '<||>'
CURSOR_TOKEN: str = '<|>'

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


def capitalize_first_letter(s: str):
    if len(s) != 0:
        return s[0].upper() if len(s) == 1 else s[0].upper() + s[1:]
    else:
        return s


def decapitalize_first_letter(s: str):
    if len(s) != 0:
        return s[0].lower() if len(s) == 1 else s[0].lower() + s[1:]
    else:
        return s


def invert_case(phrase: str) -> str:
    parr = phrase.split(' ')
    if len(parr) == 0 or len(parr[0]) == 0:
        return phrase
    if parr[0][0].islower():
        parr[0] = parr[0].capitalize()
    elif parr[0][0].isupper():
        parr[0] = parr[0].lower()
    else:
        return phrase
    return ' '.join(parr)


# returns n samples from half-normal distribution
def sample_half_normal(mu, sigma, n=None):
    return [abs(random.normalvariate(mu, sigma)) for _ in range(n)] if n is not None else abs(random.normalvariate(mu, sigma))


def string_to_tensor(s: str) -> LongTensor:
    """Convert a string to a PyTorch tensor."""
    return torch.tensor([ord(c) for c in s], dtype=torch.long)


def tensor_to_string(tensor: LongTensor) -> str:
    """Convert a PyTorch tensor back to a string."""
    return ''.join([chr(int(value)) for value in tensor])


def custom_data_collator(features) -> Dict[str, Any]:
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            elif isinstance(v, str):
                batch[k] = [f[k] for f in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

class GenPrediction(EvalPrediction):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        inputs_strs: Optional[List[Tuple[LongTensor, LongTensor]]] = None
    ):
        super().__init__(predictions, label_ids, inputs)
        self.inputs_strs = inputs_strs
        print(inputs_strs[:10])

# noinspection PyMethodMayBeStatic
class EditSamplingStrategy:

    class EditSamplingStatistics:
        def __init__(self):
            self.num_samples = 0
            self.num_inversions = 0
            self.word_count_histograms = {
                'erroneous sentences': defaultdict(int),
                'target sentences': defaultdict(int),
                'phrases': defaultdict(int),
                'corrected sentences': defaultdict(int)
            }
            self.word_count_stats = {}

        def reset(self):
            self.__init__()

        def update_stats(self):
            for k, histogram in self.word_count_histograms.items():
                if not histogram:
                    self.word_count_stats[k] = 'no histogram'
                    continue

                # to numpy arrays
                values = np.array(list(histogram.keys()))
                frequencies = np.array(list(histogram.values()))

                # mean
                mean = np.sum(values * frequencies) / np.sum(frequencies)

                # median
                cumulative_frequencies = np.cumsum(frequencies)
                median_idx = np.searchsorted(cumulative_frequencies, np.sum(frequencies) / 2)
                median = values[median_idx]

                # variance & standard deviation
                variance = np.sum(frequencies * (values - mean) ** 2) / np.sum(frequencies)
                std_dev = np.sqrt(variance)
                self.word_count_stats[k] = {'mean': mean, 'median': median, 'variance': variance, 'std dev': std_dev}

        def __str__(self):
            if not self.word_count_stats:
                self.update_stats()
            return f'total samples drawn: {self.num_samples}\ntotal letter case inversions: {self.num_inversions}\nword count stats: {self.word_count_stats}'

    def __init__(self,
                 correction_strategy: str = 'uniform-multiple',
                 correction_distrib: Tuple = (),
                 cursor_strategy: str = 'none',
                 cursor_relax: int = 5,
                 cursor_relax_unit: str = 'characters',
                 cursor_rep: str = 'token',
                 log_stuff: bool = False,
                 invert_case_prob: float = None,
                 inference = False,
                 postprocessor = None,
                 **kwargs):
        self.log_stuff = log_stuff
        self.cursor_relax = cursor_relax
        self.invert_case_prob = invert_case_prob
        self.correction_distrib = correction_distrib
        self.correction_strategy = correction_strategy
        self.cursor_strategy = cursor_strategy
        self.cursor_rep = cursor_rep

        if correction_strategy == 'minimum-multiple':
            self.correct = self.correct_minimum_multiple
        elif correction_strategy == 'uniform-multiple':
            self.correct = self.correct_uniform_multiple
        elif correction_strategy == 'uniform-single':
            self.correct = self.correct_uniform_single
        elif correction_strategy == 'normal-multiple':
            self.correct = self.correct_normal_multiple
        elif correction_strategy == 'normal-single':
            self.correct = self.correct_normal_single
        else:
            raise ValueError(f"Invalid correction_strategy '{correction_strategy}'")

        if cursor_strategy == 'uniform':
            self.add_cursor = self.add_cursor_uniform
        elif cursor_strategy == 'normal':
            self.add_cursor = self.add_cursor_normal
        elif cursor_strategy == 'none' or cursor_strategy is None:
            self.add_cursor = self.add_cursor_none
        elif cursor_strategy == 'edge':
            pass
        else:
            raise ValueError(f"Invalid cursor_strategy '{cursor_strategy}'")

        if cursor_relax_unit in ['characters', 'words']:
            self.cursor_relax_unit = cursor_relax_unit
        else:
            raise ValueError(f"Invalid cursor relax unit '{cursor_relax_unit}'")

        if self.invert_case_prob is not None and (self.invert_case_prob < 0 or self.invert_case_prob > 1):
            raise ValueError(f"Invalid letter case inversion probability '{self.invert_case_prob}'")
        
        self.postprocesser = postprocessor
        if postprocessor is None:
            self.postprocess_inner = self.postprocess_none
        elif postprocessor == 'extend-sentences':
            self.postprocess_inner = self.postprocess_extend
        
        if inference:
            self.postprocess = self.postprocess_inference
        else:
            self.postprocess = self.postprocess_train

        for attr, value in kwargs.items():
            self.__setattr__(attr, value)

        self.statistics = self.EditSamplingStatistics()

    def get_stats(self):
        self.statistics.update_stats()
        return self.statistics

    def correct_minimum_multiple(self, e_arr, t_arr, edits):
        c_arr = e_arr.copy()
        edits_made = []

        rand_edit_idx = random.randrange(0, len(edits))
        rand_edit = edits[rand_edit_idx]

        # generate a correction phrase
        left_b, right_b = 0, len(t_arr)
        required_context = 1 if rand_edit[1][0] == rand_edit[1][1] else 0  # context for deletion error
        p_l, p_r = max(left_b, rand_edit[1][0] - required_context), min(right_b, rand_edit[1][1] + required_context)
        phrase = ' '.join(t_arr[p_l:p_r])  # final generated phrase

        c_arr = c_arr[:rand_edit[0][0]] + t_arr[rand_edit[1][0]:rand_edit[1][1]] + c_arr[rand_edit[0][1]:]
        edits_made.append(rand_edit)

        return ' '.join(c_arr), phrase, edits_made


    # at least one error will be corrected
    # 1. a random error is chosen. this error must be corrected
    # 2. a correction phrase is generated with a uniform distribution across ALL indices
    # 3. all edits that correspond to errors that are completely contained within the correction phrase will be applied
    def correct_uniform_multiple(self, e_arr, t_arr, edits):
        c_arr = e_arr.copy()  # corrected sentence is initially identical to the error sentence. it will be corrected differently based on the 'correction_strategy'
        # parameter in the branches below.
        edits_made = []  # list of edits that were utilized (used in generating the cursor location)

        # choose a random edit to be included in the edits made
        rand_edit_idx = random.randrange(0, len(edits))
        rand_edit = edits[rand_edit_idx]

        # generate a correction phrase
        left_b, right_b = 0, len(t_arr)
        required_context = 1 if rand_edit[1][0] == rand_edit[1][1] else 0  # context for deletion error
        rand_l, rand_r = random.randint(left_b, max(left_b, rand_edit[1][0] - required_context)), random.randint(
            min(right_b, rand_edit[1][1] + required_context),
            right_b)  # sampled indices

        # apply the edit(s)
        for i in range(len(edits) - 1, -1, -1):
            cur_edit = edits[i]
            required_context = 1 if cur_edit[1][0] == cur_edit[1][1] else 0  # context for deletion error
            context_left, context_right = max(left_b, cur_edit[1][0] - required_context), min(right_b, cur_edit[1][1] + required_context)

            if context_right > rand_r:
                rand_r = min(context_left + required_context, rand_r)
                continue
            if context_left < rand_l:
                rand_l = max(context_right - required_context, rand_l)
                break

            # apply edit
            c_arr = c_arr[:cur_edit[0][0]] + t_arr[cur_edit[1][0]:cur_edit[1][1]] + c_arr[cur_edit[0][1]:]

            # save the edits applied
            edits_made.append(cur_edit)

        phrase = ' '.join(t_arr[rand_l:rand_r])  # final generated phrase

        return ' '.join(c_arr), phrase, edits_made

    # exactly one randomly chosen error will be corrected, and the correction phrase must cover and only cover that error
    # 1. a random error is chosen. this error will be the only one corrected
    # 2. a correction phrase is generated with a uniform distribution bounded by the end of the previous error and the beginning of the next error (if any)
    # 3. the edit corresponding to the error will be applied
    def correct_uniform_single(self, e_arr, t_arr, edits):
        c_arr = e_arr.copy()  # corrected sentence is initially identical to the error sentence. it will be corrected differently based on the 'correction_strategy'
        # parameter in the branches below.
        edits_made = []  # list of edits that were utilized (used in generating the cursor location)

        # choose a random edit to make
        rand_edit_idx = random.randrange(0, len(edits))
        rand_edit = edits[rand_edit_idx]

        # generate a correction phrase
        left_b, right_b = 0 if rand_edit_idx == 0 else edits[rand_edit_idx - 1][1][1], len(t_arr) if rand_edit_idx == len(edits) - 1 else \
            edits[rand_edit_idx + 1][1][0]
        required_context = 1 if rand_edit[1][0] == rand_edit[1][1] else 0  # context for deletion error
        rand_l, rand_r = random.randint(left_b, max(left_b, rand_edit[1][0] - required_context)), random.randint(
            min(right_b, rand_edit[1][1] + required_context),
            right_b)  # sampled indices

        phrase = ' '.join(t_arr[rand_l:rand_r])  # final generated phrase

        # apply the edit
        c_arr = c_arr[:rand_edit[0][0]] + t_arr[rand_edit[1][0]:rand_edit[1][1]] + c_arr[rand_edit[0][1]:]

        # save the edit applied
        edits_made.append(rand_edit)

        return ' '.join(c_arr), phrase, edits_made

    def correct_normal_multiple(self, e_arr, t_arr, edits):
        if self.correction_distrib is None:
            raise ValueError(f"Using normal distribution, so two values (mean, std dev.) are expected in correction_distrib. Got NoneType instead.")
        if len(self.correction_distrib) != 2:
            raise ValueError(
                f"Using normal distribution, so two values (mean, std dev.) are expected in correction_distrib. Got {len(self.correction_distrib)} values instead.")

        c_arr = e_arr.copy()  # corrected sentence is initially identical to the error sentence. it will be corrected differently based on 'correction_strategy'
        edits_made = []  # list of edits that were utilized (used in generating the cursor location)

        # choose a random edit to be included in the edits made
        rand_edit_idx = random.randrange(0, len(edits))
        rand_edit = edits[rand_edit_idx]

        # generate a correction phrase
        left_b, right_b = 0, len(t_arr)
        required_context = 1 if rand_edit[1][0] == rand_edit[1][1] else 0
        dist_l, dist_r = sample_half_normal(*self.correction_distrib, 2)
        dist_l, dist_r = round(dist_l), round(dist_r)
        rand_l, rand_r = max(left_b, rand_edit[1][0] - required_context - dist_l), min(right_b, rand_edit[1][1] + required_context + dist_r)

        # apply the edit(s)
        for i in range(len(edits) - 1, -1, -1):
            cur_edit = edits[i]
            required_context = 1 if cur_edit[1][0] == cur_edit[1][1] else 0  # context for deletion error
            context_left, context_right = max(left_b, cur_edit[1][0] - required_context), min(right_b, cur_edit[1][1] + required_context)

            if context_right > rand_r:
                rand_r = min(context_left + required_context, rand_r)
                continue
            if context_left < rand_l:
                rand_l = max(context_right - required_context, rand_l)
                break

            # apply edit
            c_arr = c_arr[:cur_edit[0][0]] + t_arr[cur_edit[1][0]:cur_edit[1][1]] + c_arr[cur_edit[0][1]:]

            # save the edits applied
            edits_made.append(cur_edit)

        phrase = ' '.join(t_arr[rand_l:rand_r])  # final generated phrase

        return ' '.join(c_arr), phrase, edits_made

    def correct_normal_single(self, e_arr, t_arr, edits):
        if self.correction_distrib is None:
            raise ValueError(f"Using normal distribution, so two values (mean, std dev.) are expected in correction_distrib. Got NoneType instead.")
        if len(self.correction_distrib) != 2:
            raise ValueError(
                f"Using normal distribution, so two values (mean, std dev.) are expected in correction_distrib. Got {len(self.correction_distrib)} values instead.")

        c_arr = e_arr.copy()  # corrected sentence is initially identical to the error sentence. it will be corrected differently based on the 'correction_strategy'
        # parameter in the branches below.
        edits_made = []  # list of edits that were utilized (used in generating the cursor location)

        # choose a random edit to make
        rand_edit_idx = random.randrange(0, len(edits))
        rand_edit = edits[rand_edit_idx]

        # generate a correction phrase
        left_b, right_b = 0 if rand_edit_idx == 0 else edits[rand_edit_idx - 1][1][1], len(t_arr) if rand_edit_idx == len(edits) - 1 else \
            edits[rand_edit_idx + 1][1][0]
        required_context = 1 if rand_edit[1][0] == rand_edit[1][1] else 0
        dist_l, dist_r = sample_half_normal(*self.correction_distrib, 2)
        dist_l, dist_r = round(dist_l), round(dist_r)
        rand_l, rand_r = max(left_b, rand_edit[1][0] - required_context - dist_l), min(right_b, rand_edit[1][1] + required_context + dist_r)
        phrase = ' '.join(t_arr[rand_l:rand_r])  # final generated phrase

        # apply the edit
        c_arr = c_arr[:rand_edit[0][0]] + t_arr[rand_edit[1][0]:rand_edit[1][1]] + c_arr[rand_edit[0][1]:]

        # save the edit applied
        edits_made.append(rand_edit)

        return ' '.join(c_arr), phrase, edits_made

    # no cursor will be added
    def add_cursor_none(self, e_arr, edits_made):
        return ' '.join(e_arr)

    def add_cursor_left(self, e_arr, edits_made):
        e_str = " ".join(e_arr)

        # choose random applied edit and get the start&end of its error
        rand_edit_idx = random.randrange(0, len(edits_made))
        e_left = edits_made[rand_edit_idx][0][0]

        # choose cursor location
        if self.cursor_relax is None:
            left_b = 0
        elif self.cursor_relax_unit == 'words':
            raise NotImplementedError()
        else:
            # Add a 5-character margin
            str_l = sum(len(word) + 1 for word in e_arr[:e_left])
            left_b = min(len(e_str), max(0, str_l - self.cursor_relax))

        rand_loc = left_b

        # snap to nearest space
        space_l = rand_loc
        if space_l >= len(e_str) and self.log_stuff:
            print('space_l >= len(e_str)')
        while space_l != -1 and space_l < len(e_str) and e_str[space_l] != ' ':
            space_l -= 1

        space_r = rand_loc
        if space_r > len(e_str) and self.log_stuff:
            print('space_r > len(e_str)')
        if space_r < 0 and self.log_stuff:
            print('space_r < 0')
        while space_r != len(e_str) and e_str[space_r] != ' ':
            space_r += 1

        sdist_l, sdist_r = rand_loc - space_l, space_r - rand_loc
        if sdist_r < sdist_l:
            e_str = f'{e_str[0:space_r]} {CURSOR_TOKEN}{e_str[space_r:]}'
        else:
            if space_l == -1:
                e_str = f'{CURSOR_TOKEN} {e_str}'
            else:
                e_str = f'{e_str[0:space_l]} {CURSOR_TOKEN}{e_str[space_l:]}'
        return e_str

    def add_cursor_right(self, e_arr, edits_made):
        e_str = " ".join(e_arr)

        # choose random applied edit and get the start&end of its error
        rand_edit_idx = random.randrange(0, len(edits_made))
        e_right = edits_made[rand_edit_idx][0][1]

        # choose cursor location
        if self.cursor_relax is None:
            left_b, right_b = (0, len(e_str))
        elif self.cursor_relax_unit == 'words':
            raise NotImplementedError()
        else:
            # Add a 5-character margin
            str_r = sum(len(word) + 1 for word in e_arr[:e_right]) - 1
            right_b = min(len(e_str), str_r + self.cursor_relax)

        rand_loc = right_b

        # snap to nearest space
        space_l = rand_loc
        if space_l >= len(e_str) and self.log_stuff:
            print('space_l >= len(e_str)')
        while space_l != -1 and space_l < len(e_str) and e_str[space_l] != ' ':
            space_l -= 1

        space_r = rand_loc
        if space_r > len(e_str) and self.log_stuff:
            print('space_r > len(e_str)')
        if space_r < 0 and self.log_stuff:
            print('space_r < 0')
        while space_r != len(e_str) and e_str[space_r] != ' ':
            space_r += 1

        sdist_l, sdist_r = rand_loc - space_l, space_r - rand_loc
        if sdist_r < sdist_l:
            e_str = f'{e_str[0:space_r]} {CURSOR_TOKEN}{e_str[space_r:]}'
        else:
            if space_l == -1:
                e_str = f'{CURSOR_TOKEN} {e_str}'
            else:
                e_str = f'{e_str[0:space_l]} {CURSOR_TOKEN}{e_str[space_l:]}'
        return e_str

    # - uniform probability to choose any edit that was applied
    # - uniform probability for any cursor location within an interval centered on the error of the edit chosen
    def add_cursor_uniform(self, e_arr, edits_made):
        e_str = " ".join(e_arr)

        # choose random applied edit and get the start&end of its error
        rand_edit_idx = random.randrange(0, len(edits_made))
        e_left, e_right = edits_made[rand_edit_idx][0]

        # choose cursor location
        if self.cursor_relax is None:
            left_b, right_b = (0, len(e_str))
        elif self.cursor_relax_unit == 'words':
            raise NotImplementedError()
        else:
            # Add a 5-character margin
            str_l, str_r = sum(len(word) + 1 for word in e_arr[:e_left]), sum(len(word) + 1 for word in e_arr[:e_right]) - 1
            left_b = min(len(e_str), max(0, str_l - self.cursor_relax))
            right_b = min(len(e_str), str_r + self.cursor_relax)

        if left_b > right_b and self.log_stuff:
            print('left_b > right_b')
        right_b = max(left_b, right_b)
        rand_loc = random.randint(left_b, right_b)

        # snap to nearest space
        space_l = rand_loc
        if space_l >= len(e_str) and self.log_stuff:
            print('space_l >= len(e_str)')
        while space_l != -1 and space_l < len(e_str) and e_str[space_l] != ' ':
            space_l -= 1

        space_r = rand_loc
        if space_r > len(e_str) and self.log_stuff:
            print('space_r > len(e_str)')
        if space_r < 0 and self.log_stuff:
            print('space_r < 0')
        while space_r != len(e_str) and e_str[space_r] != ' ':
            space_r += 1

        sdist_l, sdist_r = rand_loc - space_l, space_r - rand_loc
        if sdist_r < sdist_l:
            e_str = f'{e_str[0:space_r]} {CURSOR_TOKEN}{e_str[space_r:]}'
        else:
            if space_l == -1:
                e_str = f'{CURSOR_TOKEN} {e_str}'
            else:
                e_str = f'{e_str[0:space_l]} {CURSOR_TOKEN}{e_str[space_l:]}'
        return e_str

    # - uniform probability to choose any edit that was applied
    # - gaussian sampling for cursor location, centered on the error of the edit chosen
    def add_cursor_normal(self, e_arr, edits_made):
        e_str = " ".join(e_arr)

        # choose random applied edit and get the start&end of its error
        rand_edit_idx = random.randrange(0, len(edits_made))
        e_left, e_right = edits_made[rand_edit_idx][0]

        # choose cursor location
        if self.cursor_relax is None:
            raise NotImplementedError()
        elif self.cursor_relax_unit == 'words':
            raise NotImplementedError()
        else:
            # sample cursor location
            str_l, str_r = sum(len(word) + 1 for word in e_arr[:e_left]), sum(len(word) + 1 for word in e_arr[:e_right]) - 1
            str_mid = (str_l + str_r) / 2
            rand_loc = min(len(e_str), max(0, (int)(0.5 + random.normalvariate(str_mid, self.cursor_relax))))

        # snap to nearest space
        space_l = rand_loc
        if space_l >= len(e_str) and self.log_stuff:
            print('space_l >= len(e_str)')
        while space_l != -1 and space_l < len(e_str) and e_str[space_l] != ' ':
            space_l -= 1

        space_r = rand_loc
        if space_r > len(e_str) and self.log_stuff:
            print('space_r > len(e_str)')
        if space_r < 0 and self.log_stuff:
            print('space_r < 0')
        while space_r != len(e_str) and e_str[space_r] != ' ':
            space_r += 1

        sdist_l, sdist_r = rand_loc - space_l, space_r - rand_loc
        if sdist_r < sdist_l:
            e_str = f'{e_str[0:space_r]} {CURSOR_TOKEN}{e_str[space_r:]}'
        else:
            if space_l == -1:
                e_str = f'{CURSOR_TOKEN} {e_str}'
            else:
                e_str = f'{e_str[0:space_l]} {CURSOR_TOKEN}{e_str[space_l:]}'
        return e_str

    def postprocess_none(self, e_str, sentence, phrase, c_str, tok):
        input_str = f'{e_str} {TEXT_SEP_TOKEN} {phrase}'
        label_str = c_str
        return self.apply_cursor_rep(input_str, label_str, tok)

    def postprocess_extend(self, e_str, sentence, phrase, c_str, tok, **kwargs):
        if not hasattr(self, 'extension_strategy'):
            raise ValueError('no sentence extension strategy provided for postprocess_extend')

        sentences_before = kwargs['sentences_before']
        sentences_after = kwargs['sentences_after']
        sentences_before = split_into_sentences(sentences_before)
        sentences_after = split_into_sentences(sentences_after)

        n_before = 0
        n_after = 0
        if self.extension_strategy == 'random':
            n_before = random.randint(0, len(sentences_after))
            n_after = random.randint(0, len(sentences_after))
        elif self.extension_strategy == 'left-random':
            if hasattr(self, 'left_ext_max'):
                left_ext_max = self.left_ext_max
            else:
                left_ext_max = len(sentences_before)
            n_before = random.randint(0, left_ext_max)
            n_after = len(sentences_after)
        else:
            raise ValueError('invalid extension_strategy')

        sentences_before = ' '.join(sentences_before[-n_before:]) if n_before != 0 else ''
        sentences_after = ' '.join(sentences_after[:n_after])

        input_str = e_str
        input_str = f'{sentences_before} {input_str} {sentences_after}'
        if 'extend_label' in kwargs and kwargs['extend_label']:
            label_str = f'{sentences_before} {c_str} {sentences_after}'
        else:
            label_str = f'{c_str}'
        
        input_str = f'{input_str} {TEXT_SEP_TOKEN} {phrase}'

        return self.apply_cursor_rep(input_str, label_str, tok)

    def postprocess_train(self, e_str, sentence, phrase, c_str, tok, **kwargs):
        return self.postprocess_inner(e_str, sentence, phrase, c_str, tok, **kwargs)

    def postprocess_inference(self, e_str, sentence, phrase, c_str, tok, **kwargs):
        item = self.postprocess_inner(e_str, sentence, phrase, c_str, tok, **kwargs)
        item['sentence'] = sentence
        item['part'] = phrase
        return item
    
    def apply_cursor_rep(self, input_str, label_str, tok):
        if self.cursor_rep == 'naive':
            return self.apply_naive_cursor(input_str, label_str, tok)
        elif self.cursor_rep == 'mask':
            return self.apply_mask_cursor(input_str, label_str, tok)
        elif self.cursor_rep == 'token':
            return self.apply_mask_cursor(input_str, label_str, tok)
        else:
            raise ValueError(f'invalid cursor representation {self.cursor_rep}')

    def apply_mask_cursor(self, input_str, label_str, tok):
        input_tokenized = tok.tokenize(input_str)
        cursor_tok_loc = input_tokenized.index(CURSOR_TOKEN)

        input_arr = input_str.split(' ')
        input_arr.remove(CURSOR_TOKEN)
        input_str = ' '.join(input_arr)

        source = tok(input_str, padding='max_length', truncation=True, return_tensors='pt',
                                max_length=MAX_SOURCE_TOKENS)

        label = tok(label_str, padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_NEW_TOKENS)

        cursor_mask = torch.zeros(size=source['input_ids'].squeeze().size())
        cursor_mask[cursor_tok_loc] = cursor_mask[cursor_tok_loc - 1] = 1

        item = {
            "input_ids": source['input_ids'].squeeze(),
            "labels": label['input_ids'].squeeze(),

            "attention_mask": source['attention_mask'].squeeze(),
            "decoder_attention_mask": label['attention_mask'].squeeze(),
            "cursor_mask": cursor_mask,

            "input_str": input_str,
            "label_str": label_str
        }

        item["labels"] = [-100 if token == tok.pad_token_id else token for token in
                          item["labels"]]  # we do not wish to include pad tokens when calculating loss
        return item

    def apply_naive_cursor(self, input_str, label_str, tok):
        input_arr = input_str.split(' ')
        cursor_loc = input_arr.index(CURSOR_TOKEN)
        input_arr.remove(CURSOR_TOKEN)

        input_str = ' '.join([f'Error location: {cursor_loc}', *input_arr])

        source = tok(input_str, padding='max_length', truncation=True, return_tensors='pt',
                                max_length=MAX_SOURCE_TOKENS)

        label = tok(label_str, padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_NEW_TOKENS)
        
        
        item = {
            "input_ids": source['input_ids'].squeeze(),
            "labels": label['input_ids'].squeeze(),

            "attention_mask": source['attention_mask'].squeeze(),
            "decoder_attention_mask": label['attention_mask'].squeeze(),

            "input_str": input_str,
            "label_str": label_str
        }

        item["labels"] = [-100 if token == tok.pad_token_id else token for token in
                          item["labels"]]  # we do not wish to include pad tokens when calculating loss
        return item
    
    def apply_token_cursor(self, input_str, label_str, tok):
        source = tok(input_str, padding='max_length', truncation=True, return_tensors='pt',
                                max_length=MAX_SOURCE_TOKENS)

        label = tok(label_str, padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_NEW_TOKENS)
        
        
        item = {
            "input_ids": source['input_ids'].squeeze(),
            "labels": label['input_ids'].squeeze(),

            "attention_mask": source['attention_mask'].squeeze(),
            "decoder_attention_mask": label['attention_mask'].squeeze(),

            "input_str": input_str,
            "label_str": label_str
        }

        item["labels"] = [-100 if token == tok.pad_token_id else token for token in
                          item["labels"]]  # we do not wish to include pad tokens when calculating loss
        return item

    def filter(self, df: pd.DataFrame):
        df_l_dict = []
        df_r_dict = []
        if self.cursor_strategy == 'edge':
            for row_idx in range(len(df)):
                cur_row = df.iloc[row_idx]

                e_str, t_str, edits = cur_row['error_sentence'], cur_row['target_sentence'], cur_row['edits']
                e_arr = e_str.split(' ')

                left_valid = []
                right_valid = []
                for edit in edits:
                    e_left, e_right = edit[0]

                    if self.cursor_relax_unit == 'characters':
                        # check if left is valid
                        e_left_ch_idx = sum(len(word) + 1 for word in e_arr[:e_left]) - self.cursor_relax
                        if e_left_ch_idx >= 0:
                            left_valid.append(edit)

                        # check if right is valid
                        e_right_ch_idx = sum(len(word) + 1 for word in e_arr[:e_right]) - 1 + self.cursor_relax
                        if e_right_ch_idx <= len(e_str):
                            right_valid.append(edit)
                    else:
                        NotImplementedError()

                if not left_valid:
                    if self.log_stuff:
                        print('left_valid is empty')
                else:
                    df_l_dict.append({'error_sentence': e_str, 'target_sentence': t_str, 'edits': left_valid})

                if not right_valid:
                    if self.log_stuff:
                        print('right_valid is empty')
                else:
                    df_r_dict.append({'error_sentence': e_str, 'target_sentence': t_str, 'edits': right_valid})

            return pd.DataFrame(df_l_dict, columns=['error_sentence', 'target_sentence', 'edits']), pd.DataFrame(df_r_dict, columns=['error_sentence', 'target_sentence', 'edits'])
        return df

    def strategy(self, i_sentence, t_sentence, edits: List[Tuple[Tuple[int, int], Tuple[int, int]]], **kwargs):
        self.statistics.num_samples += 1

        sentence = f'{i_sentence}'

        e_arr, t_arr = i_sentence.split(' '), t_sentence.split(' ')
        self.statistics.word_count_histograms['erroneous sentences'][len(e_arr)] += 1
        self.statistics.word_count_histograms['target sentences'][len(t_arr)] += 1

        l_sentence, i_phrase, edits_made = self.correct(e_arr=e_arr, t_arr=t_arr, edits=edits)
        self.statistics.word_count_histograms['corrected sentences'][len(l_sentence.split(' '))] += 1

        if self.cursor_strategy == 'edge':
            if 'side' not in kwargs:
                ValueError("must pass 'side' param")
            if kwargs['side'] == 'left':
                self.add_cursor = self.add_cursor_left
            elif kwargs['side'] == 'right':
                self.add_cursor = self.add_cursor_right
            else:
                ValueError("invalid side")
        i_sentence = self.add_cursor(e_arr, edits_made)

        if self.log_stuff:
            print('\n')
            print(f'e: {e_arr}')
            print(f't: {t_arr}')
            print(f'edits: {edits}')
            print(f'made edits: {edits_made}')
            print((i_sentence, i_phrase, l_sentence))

        if self.invert_case_prob is not None:
            if random.random() <= self.invert_case_prob:
                i_phrase = invert_case(i_phrase)
                self.statistics.num_inversions += 1

        self.statistics.word_count_histograms['phrases'][len(i_phrase.split(' '))] += 1

        return i_sentence, sentence, i_phrase, l_sentence

    def __call__(self, e_str, t_str, edits, tok, **kwargs):
        return self.postprocess(*self.strategy(e_str, t_str, edits, **kwargs), tok, **kwargs)


class StaticCorrectionDatasetWithEdits(Dataset):
    def __init__(self, 
                 data_path,
                 tokenizer,
                 cutoff=-1,
                 inference=False,
                 log_stuff=False):
        print(f'loading data from {data_path} ... \n')
        
        self.cutoff = cutoff
        self.inference = inference
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.log_stuff = log_stuff

    def __len__(self):
        if self.cutoff != -1:
            return self.cutoff
        return len(self.df)
    
    def __getitem__(self, idx):
        tok = self.tokenizer
        row = self.df.iloc[idx]
        input_str = row['input']
        source = tok(input_str, padding='max_length', truncation=True, return_tensors='pt',
                                max_length=MAX_SOURCE_TOKENS)

        label = tok(row['label'], padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_NEW_TOKENS)

        item = {
            "input_ids": source['input_ids'].squeeze(),
            "labels": label['input_ids'].squeeze(),

            "attention_mask": source['attention_mask'].squeeze(),
            "decoder_attention_mask": label['attention_mask'].squeeze(),

            "input_str": input_str,
            "label_str": row['label']
        }
        
        if self.inference:
            item['sentence'] = row['input'].split(f' {TEXT_SEP_TOKEN} ')[0]
            item['part'] = row['input'].split(f' {TEXT_SEP_TOKEN} ')[1]
            if self.log_stuff:
                print(item)

        item["labels"] = [-100 if token == tok.pad_token_id else token for token in
                          item["labels"]]  # we do not wish to include pad tokens when calculating loss
        return item

class CorrectionDatasetWithEdits(Dataset):

    def __init__(self,
                 data_path,
                 tokenizer,
                 sampling_strategy: EditSamplingStrategy,
                 scale=1,
                 random_state=None,
                 save_dataset=False,
                 one_draw=False
                 ):
        print(f'preparing data from {data_path} ... \n\t', end='')

        print('reading data ... ', end='')
        self.df = pd.read_csv(data_path)

        print('parsing literals ... ', end='')
        self.df['edits'] = self.df['edits'].apply(ast.literal_eval)

        print('sampling ... ')
        self.df = self.df.sample(frac=scale, random_state=random_state).reset_index(drop=True)

        # set tokenizer
        self.tokenizer = tokenizer
        self.sampling_strategy = sampling_strategy

        if sampling_strategy.cursor_strategy == 'edge':
            self.df_l, self.df_r = self.sampling_strategy.filter(self.df)
        
        self.save_dataset = save_dataset
        self.one_draw = one_draw
        self.dataset_mem = [] if save_dataset else None
        self.item_mem = [] if one_draw else None

        self.is_extending = (sampling_strategy.postprocesser == 'extend-sentences')

        print('done')
            

    def __len__(self):
        return len(self.df) if self.sampling_strategy.cursor_strategy != 'edge' else len(self.df_l) + len(self.df_r)

    def __getitem__(self, idx):
        # return remembered item
        if self.one_draw and len(self.item_mem) == len(self):
            return self.item_mem[idx]
        
        # add item, remember item, and return item
        strategy_kwargs = {}
        if self.sampling_strategy.cursor_strategy == 'edge':
            side='left' if idx < len(self.df_l) else 'right'
            strategy_kwargs['side'] = side
            if side == 'left':
                row = self.df_l.iloc[idx]
            else:
                row = self.df_r.iloc[idx - len(self.df_l)]
        else:
            row = self.df.iloc[idx]

        if self.is_extending:
            strategy_kwargs.update({'sentences_before': row['sentences_before'], 'sentences_after': row['sentences_after']})

        item = self.sampling_strategy(row['error_sentence'], row['target_sentence'], row['edits'], self.tokenizer, **strategy_kwargs)

        if self.save_dataset:
            self.dataset_mem.append({'input': item['input_str'], 'label': item['label_str']})
        if self.one_draw:
            self.item_mem.append(item)

        return item


# sequence-to-sequence trainer with early stopping
class EarlyStoppingSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
            self,
            model=None,
            args=None,
            gen_kwargs_override=None,
            data_collator=None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer=None,
            model_init=None,
            compute_metrics:Optional[Callable[[GenPrediction], Dict]] = None,
            callbacks=None,
            optimizers=(None, None),
            preprocess_logits_for_metrics=None,
            patience: int = 4,  # Number of evaluations with no improvement after which training will be stopped
            log_stuff: bool = False,
            sequential_train_data = False
    ):
        if data_collator is None:
            data_collator = custom_data_collator
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        if gen_kwargs_override is None:
            gen_kwargs_override = {}
        self.gen_kwargs_override = gen_kwargs_override
        self.best_loss = None
        self.patience = patience
        self.no_improve_steps = 0  # Counter for steps with no improvement
        self.log_stuff = log_stuff
        self.seq_train_data = sequential_train_data
        if self.log_stuff:
            print(f'trainer control: {self.control}')

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            if not self.seq_train_data:
                dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["sentence", "part"]

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ):
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        all_inputs_strs = []
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, _inputs in enumerate(dataloader):
            # set copy of inputs
            inputs = {k: v for k, v in _inputs.items()}
            all_inputs_strs += [(sentence, part) for sentence, part in zip(inputs.pop('sentence'), inputs.pop('part'))]

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=-1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=-1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=-1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and self.accelerator.sync_gradients:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    GenPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs, inputs_strs=all_inputs_strs)
                )
            else:
                metrics = self.compute_metrics(GenPrediction(predictions=all_preds, label_ids=all_labels, inputs_strs=all_inputs_strs))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        output = EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
        if metric_key_prefix == 'test':
            return output

        metric = output.metrics[f'{metric_key_prefix}_loss']
        if self.log_stuff:
            print(output)
            print(metric)

        if self.best_loss is None or self.best_loss > metric:
            self.best_loss = metric
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        if self.no_improve_steps >= self.patience:
            self.control.should_training_stop = True

        if self.log_stuff:
            print({'no_improve_steps': self.no_improve_steps, 'bloss': self.best_loss, 'should_training_stop': self.control.should_training_stop})

        return output

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        gen_inputs = {k: v for k, v in inputs.items() if k not in ['decoder_attention_mask', 'labels']}
        generated_tokens = self.model.generate(**gen_inputs, **self.gen_kwargs_override)

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        gen_config = self.model.generation_config
        if generated_tokens.shape[-1] < self.gen_kwargs_override['max_new_tokens']:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.gen_kwargs_override['max_new_tokens'])
        elif generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)
        pred_group_size = 1 if 'num_return_sequences' not in self.gen_kwargs_override else self.gen_kwargs_override['num_return_sequences']
        generated_tokens = generated_tokens.reshape(generated_tokens.shape[0] // pred_group_size, pred_group_size, -1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        logs["eval_bloss"] = self.best_loss

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


class TopKAccuracy:
    def __init__(self):
        pass

    @staticmethod
    def compute(predictions: List[List[Any]], references: List[Any], k_metrics: int | List[int] = 1, strict_k: bool = True):
        num_samples = len(references)

        if not predictions:
            raise ValueError("Predictions list is empty.")

        if len(predictions) != num_samples:
            raise ValueError("Predictions and references should have the same length.")

        if isinstance(k_metrics, int):
            k_metrics = [k_metrics]

        if strict_k:
            max_k = max(k_metrics)
            if len(predictions[0]) < max_k:
                raise ValueError(f'Chosen k, {max_k}, is too large.')

        scores = {}
        for k in k_metrics:
            pred_slice = [pred[:k] for pred in predictions]
            num_correct = sum(ref in beams for beams, ref in zip(pred_slice, references))
            k_accuracy = num_correct / num_samples
            scores[f't{k}acc'] = k_accuracy

        return scores


class ClassifierHead(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, 64, device=device)
        self.lin2 = nn.Linear(64, 32, device=device)
        self.lin3 = nn.Linear(32, num_classes, device=device)

    def forward(self, x):
        out1 = F.relu(self.lin1(x))
        out2 = F.relu(self.lin2(out1))
        return F.softmax(self.lin3(out2), -1)


class T5SpeechTextClassifier(nn.Module):

    def __init__(self, base_model: str, max_source_length: int = 512, num_classes: int = 2):
        super().__init__()
        self.max_source_length = max_source_length
        self.encoder = T5EncoderModel.from_pretrained(base_model)
        self.flattener = nn.Flatten()
        self.dummy_inputs = tensor(np.zeros((1, self.max_source_length), dtype='long'))
        self.classifier_head = ClassifierHead(
            input_dim=self.flattener(self.encoder(input_ids=self.dummy_inputs, attention_mask=self.dummy_inputs).last_hidden_state).shape[1],
            num_classes=num_classes)

    def classify_text(self, text: str, tok):
        inputs = tok(text, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_source_length).to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return self.forward(input_ids=input_ids, attention_mask=attention_mask)

    def forward(self, input_ids: Optional[LongTensor], attention_mask: Optional[FloatTensor], labels: Optional[FloatTensor] = None):
        encodings = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        classifier_inputs = self.flattener(encodings)
        output_state = self.classifier_head(classifier_inputs)

        if labels is not None:
            loss = F.cross_entropy(output_state, labels)
            # print((loss, output_state, output_state.argmax(-1)))
            return {'loss': loss, 'distributions': output_state, 'predictions': output_state.argmax(-1)}
        else:
            return {'distributions': output_state, 'predictions': output_state.argmax(-1)}

if __name__ == '__main__':
    save_dir = './'  # save directory for pretty much anything that is saved by this program (models, tokenizers, logs, etc.)
    models_dir = f'{save_dir}saved_models/'
    data_dir = f'{save_dir}DELETION-INSERTION-MULTIPLE-REPLACEMENT-SINGLE/'

    base_model_name = "google/flan-t5-small"

    NUM_EPOCHS = 7  # number of epochs

    TRAIN_BATCH_SIZE = 16  # batch size when training
    VAL_BATCH_SIZE = 24  # batch size when running on validation set

    LOGGING_RATE = 0.005  # if integer, log stats every LOGGING_RATE steps. if float, log stats after every LOGGING_RATE portion of the training steps
    EVAL_RATE = 0.05  # if integer, eval every LOGGING_RATE steps. if float, eval after every LOGGING_RATE portion of the training steps
    SAVE_RATE = 0.05

    NUM_SAVES = 5

    LEARNING_RATE = 5e-5  # learning rate
    WEIGHT_DECAY = 0.001  # weight decay

    PORTION = 1  # proportion of datasets to use (note: applies to each split)

    custom_special_tokens_dict = {'additional_special_tokens': [CURSOR_TOKEN, TEXT_SEP_TOKEN]}


    def init_model() -> T5ForConditionalGeneration:
        model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        model.resize_token_embeddings(new_num_tokens=len(custom_special_tokens_dict['additional_special_tokens']))
        return T5ForConditionalGeneration.from_pretrained(base_model_name)


    def init_tokenizer() -> T5Tokenizer:
        tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        print(f"added {tokenizer.add_special_tokens(custom_special_tokens_dict)} custom special tokens to tokenizer")
        return tokenizer


    tokenizer = init_tokenizer()
    datasets = {
        'extend': EditSamplingStrategy(correction_strategy='normal-multiple', 
                                        correction_distrib=(0, 1), 
                                        cursor_strategy='normal', 
                                        cursor_relax=5, 
                                        cursor_rep='mask', 
                                        invert_case_prob=0.5, 
                                        postprocessor='extend-sentences',
                                        extension_strategy='left-random',
                                        left_ext_max=5,
                                        log_stuff=False) # baseline
    }

    for name, strategy in datasets.items():
        sampling_strategy = strategy
        # train_dataset = CorrectionDatasetWithEdits(f'{data_dir}train_data.csv', tokenizer=tokenizer, sampling_strategy=sampling_strategy, scale=PORTION)
        # val_dataset = CorrectionDatasetWithEdits(f'{data_dir}val_data.csv', tokenizer=tokenizer, sampling_strategy=sampling_strategy, scale=PORTION)
        test_dataset = CorrectionDatasetWithEdits(f'{data_dir}all_data_with_context.csv', tokenizer=tokenizer, sampling_strategy=sampling_strategy, scale=PORTION)

        print(f'|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n'
            f'|-------------------------------------------------------------------------------|\n'
            f'|-------------------------------------------------------------------------------|\n'
            f'|--------------------------- dataset: {name} -----------------------------------|\n'
            f'|--------------------------- dataset: {name} -----------------------------------|\n'
            f'|--------------------------- dataset: {name} -----------------------------------|\n'
            f'|-------------------------------------------------------------------------------|\n'
            f'|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n')
        for i in tqdm(range(0, len(test_dataset), 100)):
            item = test_dataset[i]
            input_ids = item['input_ids']
            label_ids = item['labels']
            cursor_mask = item['cursor_mask']
            idx = 0
            while idx < len(input_ids) and input_ids[idx] != 0:
                idx += 1
            input_ids = input_ids[:idx]
            idx = 0
            while idx < len(label_ids) and label_ids[idx] != -100:
                idx += 1
            label_ids = label_ids[:idx]
            in_str = tokenizer.decode(input_ids)
            la_str = tokenizer.decode(label_ids)
            if TEXT_SEP_TOKEN not in in_str:
                print('sssss')
            print(f'input string: {in_str}')
            print(f'label string: {la_str}')
            print(f'cursor mask: {cursor_mask}')
            print()
