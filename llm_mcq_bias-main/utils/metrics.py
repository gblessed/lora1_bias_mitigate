import torch
import tqdm
import pandas as pd
import time
import os
import yaml
import sys


# Add the parent directory to the path

config_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), "config/main.yaml")


with open(config_path, "r") as file:
    config = yaml.safe_load(file)


DATASET_ID = config["DATASET_ID"]
MODEL_ID = config["DATASET_ID"]

def inference_with_bias(model, tokenizer, testLoader):
    """
    Perform inference on testLoader to compute biases in option permutations for LLMs.

    Args:
        model: The language model for inference.
        tokenizer: tokenizer
        testLoader: An iterable containing test data. Each item is expected to have
                    the following keys:
                    - "inp_ids": Input IDs for the model.
                    - "inp_mask": Attention masks for the input IDs.
                    - "option_maps": A tensor mapping permutations of options, where
                      each row represents a specific permutation of option positions.
                      For example:
                      [[4, 1, 3, 2], [2, 1, 3, 4], [2, 4, 1, 3], ...]
                      This structure is used to evaluate model biases by reordering
                      the prediction probabilities to match these permutations.
    Returns:
        dict: A dictionary with "Answer_ID" and "Bias" for each example in testLoader.
    """
   

    # Initialize results dictionary
    my_ans = {"Answer_ID": [], "Bias": []}

    model.eval()  # Set model to evaluation mode
    option_ids = [tokenizer(o).input_ids[0] for o in ["1", "2", "3", "4", "5"]]
    pbar = tqdm(range(len(testLoader)), desc="Processing")

    for item in testLoader:
        if len(item["option_maps"]) > 0:
            # Process items with option maps
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    start = time.time()
                    logits = model.run_inference(item)
                    # Log processing time
                    with open(f'time_bqkv{DATASET_ID}_{MODEL_ID.split("/")[1]}.txt', 'a') as the_file:
                        the_file.write(f"{time.time() - start}\n")

            # Compute probabilities and adjust with option_maps
            preds_probabs = torch.nn.functional.softmax(logits, dim=2)[:, -1, option_ids]
            z = torch.tensor(item["option_maps"]).cpu()
            reordered_probs = torch.gather(preds_probabs.cpu(), 1, torch.argsort(z, dim=1))

            # Calculate bias as the variance across permutations
            bias = torch.mean(torch.var(reordered_probs, dim=0))

            # Resolve final prediction based on the reordered probabilities
            preds = torch.mode(z.gather(1, preds_probabs.argmax(dim=1).cpu().unsqueeze(1)).squeeze(1)).values

            # Save results
            my_ans["Answer_ID"].append(preds.item())
            my_ans["Bias"].append(bias.item())
        else:
            # Handle cases without option maps
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model.run_inference(item)
                preds = logits[:, -1, option_ids].argmax(dim=1) + 1
                my_ans["Answer_ID"].append(preds.item())

        # Save results to CSV and update progress bar
        pd.DataFrame(my_ans).to_csv("law_ans.csv", index=False)
        pbar.set_description(f"Prediction: {preds}")
        pbar.update(1)

    return my_ans

# Moved inference to model class, when  creating model we will both have model and tokenizer returned from there

# def forward_pass(model, tokenizer, batch):
#     inp_ids = batch["inp_ids"].to(model.device)
#     attn_mask = batch["inp_mask"].to(model.device)

#     if MODEL_ID != "google-t5/t5-3b":

#         decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])  # Start with <pad> token

#         result = model(input_ids=inp_ids, attention_mask=attn_mask)
#         logits = result.logits
#         return logits
#     else:
#         result = model(
#             input_ids=inp_ids,
#             decoder_input_ids=decoder_input_ids,
#             output_attentions=True,
#             output_hidden_states=True
#         )
#         logits = result.logits


   
#     return logits


