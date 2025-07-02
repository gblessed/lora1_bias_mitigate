import torch
import torch.nn as nn
import gc
import os
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
from math import ceil
from typing import Union, List, Tuple
from itertools import permutations

from utils.constants import LABEL_TO_IDX, LABELS_MAP, IDX_TO_LABEL
import numpy as np
import time

DEVICE = torch.device("cuda")


# utils
def create_prompt(data: dict) -> List[Tuple[str, str]]:
    """Creates a prompt from the given data.

    Args:
        data: A dictionary containing questions and options
    Returns:
        The prompt.
    """
    prompts = []
    prompt_format = "You are an AI assistant that answers multiple choice questions. Please respond with capitalized alphabet(s) that correspond to the correct answer {question} {options} Output: "
    for i in range(len(data["id"])):
        question = data["question"][i]
        choices = data["choices"][i]
        option_contents = choices["text"]
        option_labels = choices["label"]
        options = ""
        for j in range(len(option_contents)):
            options += f"({LABELS_MAP[option_labels[j]]}) {option_contents[j]}. "
        prompt = prompt_format.format(question=question, options=options)
        prompts.append((prompt, LABELS_MAP[data["answerKey"][i]]))
    return prompts



def create_permuted_prompts(data: dict, bias = False)-> List[List[Tuple[str, str]]]:
    prompts = []
    prompt_format = "{question}\n{options}\nOutput: "
    for i in range(len(data["id"])):
        permuted_prompts = []
        question = data["question"][i]
        question = data["question"][i]
        choices = data["choices"][i]
        option_contents = choices["text"]
        option_labels = choices["label"]
        answer_label = LABELS_MAP[data["answerKey"][i]]
        option_map = []
        option_map = [list(np.array(perm)+1) for perm in  list(permutations(range(len(data["choices"][i]["label"]))))]
        for perm in permutations(range(len(data["choices"][i]["label"]))):
            options = ""
            for j, k in enumerate(perm):
                options += f"({LABELS_MAP[option_labels[j]]}) {option_contents[k]}. "
            prompt = prompt_format.format(question=question, options=options)
            permuted_prompts.append(
                (prompt, IDX_TO_LABEL[perm.index(LABEL_TO_IDX[answer_label])])
            )
        prompts.append(permuted_prompts)
    if bias:
      return prompts, option_map
    else:
      return prompts



###### MODEL Creation #############################################################################
###################################################################################################
class Model:
    """A simple huggingface model wrapper that creates both the model and the tokenizer
    and enables access to both model and tokenizer attributes and methods.

    For example, working with the string "Which topic area is best for research" using
    Llama3-8B on gpu;
    >>> model = Model("meta-llama/Meta-Llama-3-8B", "cuda")
    >>> text = " Which topic area is best for research"
    >>> e_input = model.tokenizer_call(text, return_tensors="pt") # encode with tokenizer
    >>> outputs = model(**e_input.to(model.device)) # get_outputs
    """

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        """Initializes the model and tokenizer.

        Args:
            model_id: The huggingface model id to load.
            device: The device to load the model on. Defaults to "cuda".
        """
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            cache_dir = "/home/ubuntu/data/cache/"
        )
        self.option_to_idx = {}
        self.option_to_idx_2 = {}
        self.idx_to_option_2 ={}

        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        for label in ["A", "B", "C", "D", "E"]:
            self.option_to_idx[label] = self.hf_tokenizer.encode(label)[-1]

        for label in ["_A", "_B", "_C", "_D", "_E"]:
            self.option_to_idx_2[label] = self.hf_tokenizer.encode(label)[-1]

        self.idx_to_option = {v: k for k, v in self.option_to_idx.items()}
        self.idx_to_option_2 = {v: k for k, v in self.option_to_idx_2.items()}

        self.options = [k for k in self.idx_to_option.keys()]
        self.options_2 = [k for k in self.idx_to_option_2.keys()]

        torch.cuda.empty_cache()
        gc.collect()

    def model_call(self, *args, **kwargs):
        """Calls the huggingface model with the given arguments.

        Args:
            *args: The positional arguments to pass to the huggingface model.
            **kwargs: The keyword arguments to pass to the huggingface model.
        Returns:
            The output of the huggingface model.
        """
        return self.hf_model(*args, **kwargs)

    def tokenizer_call(self, *args, **kwargs):
        """Calls the huggingface tokenizer with the given arguments.

        Args:
            *args: The positional arguments to pass to the huggingface tokenizer.
            **kwargs: The keyword arguments to pass to the huggingface tokenizer.
        Returns:
            The output of the huggingface tokenizer.
        """
        return self.hf_tokenizer(*args, **kwargs)

    def get_answer(self, text: str, **kwargs) -> str:
        """ """
        e_input = self.tokenizer_call(text, return_tensors="pt").to(self.device)
        outputs = self.model_call(**e_input, **kwargs)
        return self.idx_to_option[
            self.options[outputs.logits[:, -1, self.options].argmax().item()]
        ]

    def get_answer_and_embedding(self, texts: List[str], **kwargs) -> str:
        """ """
        if isinstance(texts, str):
            texts = [texts]
        e_input = self.tokenizer_call(texts, return_tensors="pt").to(self.device)
        outputs = self.model_call(**e_input, **kwargs, output_hidden_states=True)
        opt_token_ids = (outputs.logits[:, -1, self.options]  + outputs.logits[:, -1, self.options_2]).argmax(dim=-1)
        result = []
        for i in range(len(texts)):
            result.append(
                (
                    self.idx_to_option[self.options[opt_token_ids[i].item()]],
                    outputs.hidden_states[-1][i, -1].detach().cpu(),
                )
            )
        return result

    def compute_bias(self, data, **kwargs)-> float:
        all_prompts, option_map = create_permuted_prompts(data, bias=True)
        e_input = self.tokenizer_call(all_prompts, return_tensors="pt").to(self.device)
        logits = self.model_call(**e_input, **kwargs).logits
        # print(logits)
        preds_probabs = torch.nn.functional.softmax(logits, dim=2)[:, -1, self.options]
        preds_probabs = preds_probabs/preds_probabs.sum(dim=1).unsqueeze(1)
        z = torch.tensor(option_map).cpu()
        reordered_probs = torch.gather(preds_probabs.cpu(), 1, torch.argsort(z, dim=1))
        bias = torch.mean(torch.var(reordered_probs, dim=0))

        return bias

    def __call__(self, *args, **kwargs):
        """Calls the huggingface model with the given arguments, ensuring that certain
        keyword arguments are set to a default value if not provided.

        Args:
            *args: The positional arguments to pass to the huggingface model.
            **kwargs: The keyword arguments to pass to the huggingface model.
        Returns:
            The output of the huggingface model.
        """
        # update keyword arguments with default values
        kwargs.update(
            {"output_hidden_states": kwargs.get("output_hidden_states", True)}
        )
        return self.model_call(*args, **kwargs)

    def __getattr__(self, name):
        """Calls the huggingface model or tokenizer attribute with the given name.

        Args:
            name: The name of the attribute to call.
        Returns:
            The output of the huggingface model or tokenizer attribute.
        """
        try:
            return getattr(self.hf_model, name)
        except AttributeError:
            return getattr(self.hf_tokenizer, name)
        raise AttributeError(f"Model has no attribute {name}")


class Llama3_8B(Model):
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    model_id = "microsoft/phi-2"


    def __init__(self, device="cuda"):
        # assert os.environ.get("HF_TOKEN", None), (
        #     "Please ensure that the HF_TOKEN environment variable is set with your"
        #     " huggingface hub token"
        # )
        # login(
        #     token=os.environ.get("HF_TOKEN")
        # )  # Loging to huggingface hub. You need to\
        # be autorized to use llama
        super().__init__(Llama3_8B.model_id, device)


######### Dataset Creation ##########################################################################
#####################################################################################################
class Dataset:
    """A simple dataset wrapper that loads the dataset from the huggingface hub."""

    def __init__(self, *args):
        """Initializes the dataset.

        Args:
            *args: The huggingface dataset id and dataset type to load.
        """
        self.data_dict = load_dataset(*args)

    def loader(self, split="train", batch_size=32):
        """Creates a loader for the dataset.

        Args:
            split: The split to load. Defaults to "train".
            batch_size: The batch size to use. Defaults to 32.
        Returns:
            A loader for the dataset.
        """

        class __cls:
            def __init__(self, data):
                self.data = data
                self._temp_data = None
                self.batch_size = batch_size
                self.split = split
                self.n = ceil(len(data) // batch_size)
                self.idx = 0

            def __next__(self):
                if self.idx >= self.n:
                    raise StopIteration
                s_idx = self.idx * self.batch_size
                e_idx = s_idx + self.batch_size
                self.idx += 1
                return self._temp_data[s_idx:e_idx]

            def __iter__(self):
                if self.split == "train":
                    self._temp_data = self.data.shuffle()
                else:
                    self._temp_data = self.data
                return self

            def __len__(self):
                return len(self.n)

            def __getitem__(self, idx):
                if isinstance(idx, int):
                    assert abs(idx) < self.n, "Index out of range"
                else:
                    raise TypeError("Index must be an integer")
                s_idx = idx * self.batch_size
                e_idx = s_idx + self.batch_size
                return self.data[s_idx:e_idx]

        return __cls(self.data_dict[split])

    def __getattr__(self, name):
        """Calls the huggingface dataset attribute with the given name.

        Args:
            name: The name of the attribute to call.
        Returns:
            The output of the huggingface dataset attribute.
        """
        try:
            return getattr(self.data_dict, name)
        except AttributeError:
            raise AttributeError(f"Dataset has no attribute {name}")


class ARCChallengeDataset(Dataset):
    dataset_id = "allenai/ai2_arc"
    dataset_type = "ARC-Challenge"

    def __init__(self):
        super().__init__(
            ARCChallengeDataset.dataset_id, ARCChallengeDataset.dataset_type
        )


###################### Bias computation for a model and dataset #####################################
#####################################################################################################
def bias_vector_computation(model, ds, k=32):
    """Computes the bias vector for a model using a given dataset
    
    Args:
        model: The model to compute the bias vector for.
        ds: The dataset to compute the bias vector for.
        k: The number of examples to consider.
    Returns:
        The bias vector.
    """
    n_examples_encountered = 0
    n_examples_considered = 0
    bias_vector = None
    done = False
    with torch.inference_mode():
        model.eval()
        for data in ds.loader(batch_size=1):
            all_prompts = create_permuted_prompts(data)
            for b_idx, perm_prompts in enumerate(all_prompts):
                n_examples_encountered += 1
                n_positive_perms = 0
                n_negative_perms = 0
                positive_bias_vector = None
                negative_bias_vector = None
                bs = 4
                n = ceil(len(perm_prompts) / bs)
                for i in range(n):
                    s_idx = i * bs
                    e_idx = s_idx + bs
                    batch_prompts = perm_prompts[s_idx:e_idx]
                    prompts = [p[0] for p in batch_prompts]
                    answers = [p[1] for p in batch_prompts]
                    res = model.get_answer_and_embedding(prompts)
                    for (pred, emb), ans in zip(res, answers):
                        if pred == ans:
                            n_positive_perms += 1
                            positive_bias_vector = (
                                0
                                if positive_bias_vector is None
                                else positive_bias_vector
                            ) + emb
                        else:
                            n_negative_perms += 1
                            negative_bias_vector = (
                                0
                                if negative_bias_vector is None
                                else negative_bias_vector
                            ) + emb
                    torch.cuda.empty_cache()
                    gc.collect()
                ratio = (
                    1e10
                    if n_negative_perms == 0
                    else n_positive_perms / n_negative_perms
                )
                if ratio >= 1 and ratio <= 2:
                    n_examples_considered += 1
                    bias_vector = (0 if bias_vector is None else bias_vector) + (
                        (1 / n_negative_perms) * negative_bias_vector
                        - (1 / n_positive_perms) * positive_bias_vector
                    )
                print(
                    f"\rNum Encountered Examples: {n_examples_encountered:3d} | Num "
                    f"Considered Examples: {n_examples_considered:3d}",
                    end="",
                )
                if n_examples_considered == k:
                    done = True
                    bias_vector = (1 / k) * bias_vector
                    break
            if done:
                break

        return bias_vector



def bias_vector_computation_dataloader(model, ds, k=32):
    """Computes the bias vector for a model using a given dataset
    
    Args:
        model: The model to compute the bias vector for.
        ds: The dataset to compute the bias vector for.
        k: The number of examples to consider.
    Returns:
        The bias vector.
    """
    n_examples_encountered = 0
    n_examples_considered = 0
    bias_vector = None
    done = False
    with torch.inference_mode():
        model.eval()
        for data in ds.loader(batch_size=1):
            all_prompts = [data["question_context"]]
            answers_all = [data["answers"]]
            for b_idx, perm_prompts in enumerate(all_prompts):
                n_examples_encountered += 1
                n_positive_perms = 0
                n_negative_perms = 0
                positive_bias_vector = None
                negative_bias_vector = None
                bs = 4
                n = ceil(len(perm_prompts) / bs)
                for i in range(n):
                    s_idx = i * bs
                    e_idx = s_idx + bs
                    batch_prompts = perm_prompts[s_idx:e_idx]
                    prompts = [p[0] for p in batch_prompts]
                    answers = [p[1] for p in batch_prompts]
                    res = model.get_answer_and_embedding(prompts)
                    for (pred, emb), ans in zip(res, answers):
                        if pred == ans:
                            n_positive_perms += 1
                            positive_bias_vector = (
                                0
                                if positive_bias_vector is None
                                else positive_bias_vector
                            ) + emb
                        else:
                            n_negative_perms += 1
                            negative_bias_vector = (
                                0
                                if negative_bias_vector is None
                                else negative_bias_vector
                            ) + emb
                    torch.cuda.empty_cache()
                    gc.collect()
                ratio = (
                    1e10
                    if n_negative_perms == 0
                    else n_positive_perms / n_negative_perms
                )
                if ratio >= 1 and ratio <= 2:
                    n_examples_considered += 1
                    bias_vector = (0 if bias_vector is None else bias_vector) + (
                        (1 / n_negative_perms) * negative_bias_vector
                        - (1 / n_positive_perms) * positive_bias_vector
                    )
                print(
                    f"\rNum Encountered Examples: {n_examples_encountered:3d} | Num "
                    f"Considered Examples: {n_examples_considered:3d}",
                    end="",
                )
                if n_examples_considered == k:
                    done = True
                    bias_vector = (1 / k) * bias_vector
                    break
            if done:
                break

        return bias_vector
###################################### Bias Node Pruning #######################################################
################################################################################################################
class wrapper:
    def __init__(self, inp):
        self.layer = inp
    # def __getattribute__(self,  key):
    #     pass
    def __getattr__(self, key):
        try:
            return getattr(self.layer, key)
        except:
            # case 1: phi-sft
            return getattr(self.layer.linear, key)

def bias_node_pruning(model, bias_vector):
    """Performs bias node pruning on a model using a given bias vector

    Args:
        model: Model to prune
        bias_vector: Bias vector to use for pruning
    Returns:
        The original weight of the model
    """
    # hf_model = model.hf_model
    hf_model = model

    with torch.no_grad():
        lm_head = wrapper(hf_model.lm_head)
        orig_weight = lm_head.weight.data.clone()
        contrib_vec = (
            bias_vector.to(lm_head.weight.data.device).contiguous().view(1, -1)
            * lm_head.weight
        ).sum(dim=0)
        mask = torch.ones_like(contrib_vec)
        mask[torch.topk(contrib_vec, 32).indices] = 0
        lm_head.weight.data = (
            lm_head.weight.data * mask.contiguous().view(1, -1)
        )
    return orig_weight


def restore_model(model, orig_weight):
    """Restores a model to its original state

    Args:
        model: Model to restore
        orig_weight: Original weight of the model
    """
    model.hf_model.lm_head.weight.data = orig_weight
