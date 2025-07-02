# Dataset.py
import os
from torch.utils.data import Dataset
from .base import load_json_file
import itertools
import math
import random 
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from models.prompt_template import get_prompt_template
class CustomDataset(Dataset):
    def __init__(self, dataset_type: str, model_type: str, tokenizer):
        """
        Custom dataset for handling multiple JSON structures.

        Args:
            json_path (str): Path to the dataset file (JSON or JSONL).
            dataset_type (str): Type of dataset (e.g., 'meqna', 'telleqna').
        """
        avail_dataset = ['teleqna',  'medmcqa' , "qasc" ]
        if dataset_type not in avail_dataset:
            print(f"Dataset must be one of {avail_dataset}")
            return 

        script_dir = os.path.dirname(os.path.abspath(__file__))

        json_file_path = os.path.join(script_dir, 'raw_json', f'{dataset_type}.json')

        self.tokenizer  = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data, self.data_map = load_json_file(dataset_type, json_file_path)

        self.dataset_type = dataset_type

        self.model_type = model_type

        self.batch_size = 1
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data point and its attributes.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: Data point with attributes as keys.
        """
        option_header = ["option 1 ", "option 2 ", "option 3 ", "option 4 ", "option 5 "]
        batch_start_id = idx * self.batch_size
        mapper_ans = {"a":1, "b":2, "c":3, "d":4, "e":5}
        batch_end_id  = min(len(self.data), batch_start_id + self.batch_size) 
        batch = {"question_context":[], "answer":[]}

        batch_type =False
        
        prompt_without_contex_train = get_prompt_template(self.dataset_type, self.model_type)

        if self.dataset_type == "medmcqa":
            for i in range(batch_start_id, batch_end_id):
                example = self.data[i]
                options = []
                opts = []
                for key in example.keys():
                    if key.startswith("op"):
                        if example[key] == None:
                            continue
                        options.append(example[key])
                        opts.append((example[key], mapper_ans[key.split("op")[1]] ))

        elif self.dataset_type == "teleqna":
            for i in range(batch_start_id, batch_end_id):
                example = self.data[self.data_map[i]]
                options = []
                opts = []
                for key in example.keys():
                    if key.startswith("opt"):
                        if example[key] == None:
                            continue
                        options.append(example[key])
                        opts.append((example[key], key.split("option ")[1]))
            
        elif self.dataset_type == "qasc":
                mapper_ans = {"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "H":8}
                option_header = ["option 1 ", "option 2 ", "option 3 ", "option 4 ", "option 5 ","option 6 ", "option 7 ", "option 8 " ]

                for i in range(batch_start_id, batch_end_id):
                    example = self.data[i]
                    options = []
                    opts = []
                    context = example["fact1"] + '\n' + example["fact2"]
                    for i, sample in enumerate(example['question']['choices']):
                        # print(key)
                        if example["answerKey"] == sample["label"]:
                            correct_option_id = mapper_ans[example["answerKey"]]
                            correct_option_txt = sample["text"]
                        
                            
                    
                        # options.append(option_header[i]+ sample['text'])
                        options.append(sample['text'])

                        opts.append((sample['text'], option_header[i].split("option")[1]))
                 
                explanation = example['combinedfact']

        string_opts = ' '.join(opt[0] for opt in opts)
        batch_prompts = []
        option_maps = []
     
        if not ("option" in string_opts or "above" in string_opts) :
            batch_type = True
            
            all_permutations = list(itertools.permutations(opts))
            for option_set in all_permutations:
                option_map = []
                options_with_header   = []
                for z in range(len(option_set)):

                    options_with_header.append(option_header[z] +option_set[z][0])

                    
                    option_map.append(int(option_set[z][1]))

                # options_with_header = "\n".join(options_with_header)
                if self.dataset_type == "medmcqa":                    
                    options_with_header = "\n".join(options_with_header)

                    prompt = prompt_without_contex_train.substitute(question = self.clean_question(example["question"]), options =options_with_header)
                
                elif self.dataset_type == "teleqna":
                    options_with_header = "\n".join(options_with_header)
                    prompt = prompt_without_contex_train.substitute(question = self.clean_question(example["question"]),\
                    abbreviation='\n'.join(example["abbreviation"]), context1 = '\n'.join(example["context_qwen2"][:2])[:512] , context2 = '\n'.join(example["context_gle"])[:512], context3 = '\n'.join(example["context_bm"][:2])[:512],
                    options =options_with_header)     
                
                elif self.dataset_type == "qasc":

                    options_with_header = "\n".join(options_with_header)

                    prompt = prompt_without_contex_train.substitute( question  = example['question']['stem'], context=  context, options = options_with_header)
                # print(question_context, prompt)
                batch_prompts.append(prompt)
                option_maps.append(option_map)
            batch["question_context"] += batch_prompts
        else:
            if self.dataset_type == "medmcqa":
            
                options_with_header = [option_header[i] +options[i] for i in range(len(options)) ]
            
                options_with_header = "\n".join(options_with_header)
                prompt = prompt_without_contex_train.substitute(question = self.clean_question(example["question"]), options =options_with_header)
            elif self.dataset_type == "teleqna":
                options_with_header = [option_header[i] +options[i] for i in range(len(options)) ]
                options_with_header = "\n".join(options_with_header)
                prompt = prompt_without_contex_train.substitute(question = self.clean_question(example["question"]),\
                abbreviation='\n'.join(example["abbreviation"]), context1 = '\n'.join(example["context_qwen2"][:2])[:512] , context2 = '\n'.join(example["context_gle"])[:512], context3 = '\n'.join(example["context_bm"][:2])[:512],
                options =options_with_header)

                # if MODEL_ID == "google-t5/t5-3b":
                #     prompt = prompt_without_contex_train.substitute(question = clean_question(example["question"]),\
                #     abbreviation='\n'.join(example["abbreviation"]), context1 = '\n'.join(example["context_qwen2"][:2])[:512] , context2 = '\n'.join(example["context_gle"])[:50], context3 = '\n'.join(example["context_bm"][:2])[:50],
                #     options =options_with_header)     
            
            elif self.dataset_type == "qasc" :
                correct_option_id = correct_option_id
                correct_option_txt_header = str(correct_option_id) + " " + correct_option_txt
                options_with_header = [option_header[i] +options[i] for i in range(len(options)) ]
                options_with_header = "\n".join(options_with_header)
                prompt = prompt_without_contex_train.substitute( question  = example['question']['stem'], context=  context, options = options_with_header)

            batch_prompts.append(prompt)

            batch["question_context"] += batch_prompts

        self.tokenizer.padding_side = "left"
        q_tokens = self.tokenizer(batch["question_context"], padding="longest", return_tensors="pt")  
        self.tokenizer.padding_side = "right"
        # a_tokens = self.tokenizer(batch["answer"], padding="longest", return_tensors="pt")
        tokens = q_tokens
        attn_masks = q_tokens["attention_mask"]
        
       

        # attn_masks = torch.cat([q_tokens["attention_mask"], a_tokens["attention_mask"]], dim=1)
        # loss_mask = torch.cat([torch.zeros_like(q_tokens["attention_mask"]), a_tokens["attention_mask"]], dim=1)[:,1:]
   
        result = {
        "inp_ids":tokens["input_ids"],
        "inp_mask":attn_masks,## Causal Training
        "option_maps": option_maps
        }

        # result["loss_mask"] = loss_mask * result["out_mask"]
        # result["out_ids"][:,:q_tokens["input_ids"].size(1)-10] = self.tokenizer.eos_token_id

        return result     

    def  clean_question(self, question):
        for num in [14, 15, 16, 17, 18]:
            question = question.replace(f"[3GPP Release {num}]", "")
        return question
    
    def  custom_collate_fn(batch,k_limit):

        def collate_fn(batch):
            """
            Custom collate function to merge a list of permuted questions into a batch.
            :param batch: List of lists of permuted question dictionaries.
            :return: Batch of permuted questions.
            """
            
            batch = batch[0]
            batch_options = batch['option_maps']
            input_ids = batch['inp_ids']
            if  len(batch_options) != 0:
                sampled_indices = random.sample(range(len(batch_options)), min(k_limit, len(batch_options)))
                batch_options = [batch_options[i] for i in sampled_indices]
                input_ids = [input_ids[i] for i in sampled_indices]
            
            batch['option_maps'] = batch_options
            batch['inp_ids'] = input_ids
            
            return batch
        return collate_fn