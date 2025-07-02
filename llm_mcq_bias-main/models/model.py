

import torch
from .base import load_huggingface_model

class Model:
    def __init__(self, model_id: str):
        """
        Initialize the Hugging Face model and tokenizer.

        Args:
            model_id (str): Hugging Face model identifier.
        """
        self.model_id = model_id
        self.model, self.tokenizer = load_huggingface_model(model_id)
        self.model.eval()  # Set the model to evaluation mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
    def run_inference(self, batch):
        inp_ids = batch["inp_ids"].to(self.device)
        attn_mask = batch["inp_mask"].to(self.device)

        if self.model_id != "google-t5/t5-3b":

            decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]])  # Start with <pad> token

            result = self.model(input_ids=inp_ids, attention_mask=attn_mask)
            logits = result.logits
            return logits
        else:
            result = self.model(
                input_ids=inp_ids,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
                output_hidden_states=True
            )
            logits = result.logits


    
        return logits

    def encode(self, batch_in):

        return self.tokenizer(batch_in, return_tensors='pt', return_attention_mask=False)

    def decode(self, batch_out):
        return self.tokenizer.batch_decode(batch_out)

    def  generate_with_text(self, batch_text, max_len):
        
        inputs = self.tokenizer(batch_text, return_tensors='pt', return_attention_mask=False).to(self.device)
        logits = self.model.generate(**inputs, max_length=max_len)
        text   = self.tokenizer.batch_decode(logits)[0]
        return text

