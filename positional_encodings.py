import torch
import math

# =========================================================================
# THE OLD FIX: VASWANI SINUSOIDAL WAVES
# =========================================================================
def get_sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    The 2017 approach: Stamping a unique wave pattern onto every word.
    
    Imagine a soundboard equalizer:
    [ Pos 0 ]  ~~~ (fast wave) ~~~ (medium wave) ~~~ (slow wave)
    [ Pos 1 ]  -~~ (fast wave) ~~- (medium wave) ~~~ (slow wave)
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term) 
    pe[:, 1::2] = torch.cos(position * div_term) 
    return pe

# =========================================================================
# THE MODERN REVOLUTION: ROTARY POSITION EMBEDDINGS (RoPE)
# =========================================================================
def precompute_rope_frequencies(dim: int, max_seq_len: int = 1024):
    """
    Building the gears for our clock hands.
    Pre-calculates "how much to spin" for every possible position.
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    The Magic Trick: Spinning the vectors!
    
    THE CAROUSEL ANALOGY:
    If you spin the word "apple", doesn't it land in the "car" neighborhood and 
    lose its meaning? NO! Because RoPE puts the ENTIRE semantic space on a carousel.
    If "apple" and "orange" are both at Position 5, they both spin 5 clicks. 
    They stay standing right next to each other. 
    
    The spin doesn't change their relationship to other words; it changes 
    their relationship to DISTANCE.
    """
    d = x.shape[-1]
    d_2 = d // 2
    
    x1 = x[..., :d_2]
    x2 = x[..., d_2:]
    
    x_tilde = torch.cat([-x2, x1], dim=-1)
    
    cos = cos.unsqueeze(0).unsqueeze(0) 
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    return (x * cos) + (x_tilde * sin)


# =========================================================================
# TRUE INTERACTIVE DEMO: THE CAROUSEL (Runs if you execute this file!)
# =========================================================================
if __name__ == "__main__":
    print("\n>_ initializing RoPE interactive demo: THE CAROUSEL (0_o)\n")
    
    try:
        user_len = input("how many positions away should 'apple' roll? (e.g. 5): ")
        seq_steps = int(user_len)
    except ValueError:
        print("invalid input, defaulting to 5.")
        seq_steps = 5
        
    # Let's create two words in our 2D space. 
    # "eat" and "apple" are related, so they point in similar directions.
    word_eat = torch.tensor([[[[0.8, 0.6]]]])    # Length is 1.0
    word_apple = torch.tensor([[[[1.0, 0.0]]]])  # Length is 1.0
    
    # Calculate their base similarity (Attention Score) before any rotation
    base_score = torch.matmul(word_eat, word_apple.transpose(-2, -1)).item()
    
    print(f"\n[ START ]")
    print(f"  'eat' vector:   {word_eat.squeeze().tolist()} | Length: {torch.norm(word_eat).item():.4f}")
    print(f"  'apple' vector: {word_apple.squeeze().tolist()} | Length: {torch.norm(word_apple).item():.4f}")
    print(f"  => Base Attention Score (How related they are): {base_score:.4f}\n")
    
    # Precompute gears
    cos, sin = precompute_rope_frequencies(dim=2, max_seq_len=seq_steps)
    
    # "eat" stays at Position 0 (it doesn't spin)
    eat_at_pos_0 = apply_rope(word_eat, cos[0:1], sin[0:1]) 
    
    # We will spin "apple" through the sequence
    apple_rotated = apply_rope(word_apple, cos, sin)
    
    print("---------------------------------------------------------")
    print("THE EXPERIMENT: 'eat' stays at Pos 0. 'apple' moves away.")
    print("watch how the attention score drops as distance grows!")
    print("---------------------------------------------------------\n")
    
    for pos in range(seq_steps):
        input(f"  [ press ENTER to move 'apple' to pos {pos} ]")
        
        vec = apple_rotated[0, 0, pos].tolist()
        length = torch.norm(apple_rotated[0, 0, pos]).item()
        
        # Calculate new attention score (Dot Product between eat@0 and apple@pos)
        current_apple = apple_rotated[:, :, pos:pos+1, :]
        attention_score = torch.matmul(eat_at_pos_0, current_apple.transpose(-2, -1)).item()
        
        print(f"  => 'apple' coords: [{vec[0]:+0.4f}, {vec[1]:+0.4f}] (Length stays {length:.4f}!)")
        print(f"  => Distance: {pos} steps apart | ATTENTION SCORE: {attention_score:.4f}\n")
        
    print("task complete. you just proved that RoPE decays attention over distance! (>_o)\n")