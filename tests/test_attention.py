import torch
from transformers import AutoTokenizer, AutoModel

def inspect_attention_heads(sentence: str, model_name: str = "bert-base-uncased", layer_idx: int = 0):
    """
    Extracts and outputs how each attention head in a specified layer 
    attends across the tokens of an input sentence.
    """
    # 1. Load Tokenizer & Model with attention output enabled
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    # 2. Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # 3. Forward pass (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    layer_attentions = outputs.attentions[layer_idx][0]  # Shape: (num_heads, seq_len, seq_len)
    num_heads, seq_len, _ = layer_attentions.shape

    print(f"Input Sentence: '{sentence}'")
    print(f"Token Sequence ({seq_len}): {tokens}")
    print(f"Model: {model_name} | Layer: {layer_idx} | Total Heads: {num_heads}")
    print("=" * 65)

    # 4. Parse per-head attention weights
    for head_idx in range(num_heads):
        head_matrix = layer_attentions[head_idx]  # Shape: (seq_len, seq_len)
        print(f"\n[ Head {head_idx + 1} ]")

        for i, src_token in enumerate(tokens):
            # Find the token receiving the highest attention weight from src_token
            max_weight, max_idx = torch.max(head_matrix[i], dim=0)
            tgt_token = tokens[max_idx.item()]
            print(f"  '{src_token}' ---> strongest focus on ---> '{tgt_token}' ({max_weight.item():.1%})")

if __name__ == "__main__":
    # Test sentence
    text = "The quick brown fox jumps over the lazy dog."
    inspect_attention_heads(sentence=text, layer_idx=0)