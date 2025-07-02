

import torch

def generate_enc_dec_t5(prompt, tokenizer, model, max_length=40):
    """
    Generates text autoregressively.

    Args:
        prompt: The input prompt (string).
        max_length: The maximum length of the generated sequence.

    Returns:
    The generated text.     
    # Example prompt
    # prompt = "My name is Sarah and I live in London"  # Define your prompt
    # # prompt = "My name is Wolfgang and I live in Berlin"
    # generated_text = generate_enc_dec_t5(prompt, max_length=15)

    # print(f"Output: {generated_text}")
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])  # Start with <pad> token
    
    generated_ids = []

    for _ in range(max_length):
        with torch.no_grad(): 
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids
            )
        
        # Get the next token logits and predicted token
        logits = outputs.logits[:, -1, :]  # Use only the last step logits
        next_token_id = logits.argmax(dim=-1)  # Get the token with the highest probability
        
        generated_ids.append(next_token_id.item())  # Store the token ID
        
       
        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(0)], dim=1)
      
        if next_token_id.item() == tokenizer.eos_token_id:
        
            break
     
 
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text




def preprocess(text):
    
    text = text.lower()
    return set(text.split())


def choose_most_likely_option(options, candidate_answer):
    candidate_words = preprocess(candidate_answer)
    
    best_option = None
    max_overlap = 0
    
    for option in options:
        option_words = preprocess(option)
        overlap = len(candidate_words.intersection(option_words))
        
  
        if overlap > max_overlap:
            max_overlap = overlap
            best_option = option
            
    return best_option, options.index(best_option)+1