# base.py

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_huggingface_model(model_id: str):
    """
    Load a Hugging Face model and its tokenizer based on the model ID.

    Args:
        model_id (str): Hugging Face model identifier.

    Returns:
        tuple: Loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)
    return model, tokenizer
