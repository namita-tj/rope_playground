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
def precompute_rope_frequencies(dim: int, max_seq_len: int = 1024, base: float = 10000.0):
    """
    Building the gears for our clock hands.
    Pre-calculates "how much to spin" for every possible position.

    Returns (cos, sin), each shaped [max_seq_len, dim // 2]: one row per position,
    one column per pair of dimensions. `base` sets how slowly the slowest hand turns
    (context-extension tricks like NTK scaling work by changing it).
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE needs an even dimension to form pairs, got {dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def rope_slice(cos: torch.Tensor, sin: torch.Tensor, seq_len: int, offset: int = 0):
    """
    Picks the gears for positions [offset, offset + seq_len).

    The precomputed table covers max_seq_len positions, but a real sentence is
    usually shorter. `offset` is where the sentence starts, e.g. when generating
    one token at a time with a KV cache.
    """
    end = offset + seq_len
    if offset < 0 or end > cos.shape[0]:
        raise ValueError(
            f"positions {offset}..{end - 1} are outside the precomputed table "
            f"(max_seq_len={cos.shape[0]})"
        )
    return cos[offset:end], sin[offset:end]

# =========================================================================
# THE STRETCH: making RoPE read texts longer than it trained on
# =========================================================================
ROPE_SCALINGS = ("pi", "ntk", "yarn")


def scaled_rope_frequencies(
    dim: int,
    max_seq_len: int,
    method: str | None,
    factor: float,
    original_max_len: int,
    base: float = 10000.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
):
    """
    RoPE gears for a text `factor` times longer than the training length.

    The problem: beyond the training length, the slow clock hands point to angles
    the model has never seen. Three fixes, all of which only change the gears:

    * "pi"   Position Interpolation (Chen et al., 2023): slow EVERY hand down by
             `factor`, so position 1024 looks like position 256 did in training.
             Cost: neighbouring words also end up closer together than usual.
    * "ntk"  NTK-aware scaling (bloc97, 2023): stretch the base instead, so the
             fast hands barely change (local detail kept) and the slow hands slow
             down the most.
    * "yarn" YaRN (Peng et al., 2023): decide per hand. Hands that turn many times
             within the training length keep their speed, hands that turn less than
             once get interpolated like PI, the ones in between are blended. It also
             sharpens attention a little, because longer texts spread it thinner.

    factor <= 1 or method None gives plain RoPE. Returns (cos, sin) like
    precompute_rope_frequencies; for YaRN they include the attention temperature.
    """
    if method is None or factor <= 1:
        return precompute_rope_frequencies(dim, max_seq_len, base)
    if method not in ROPE_SCALINGS:
        raise ValueError(f"unknown RoPE scaling {method!r}, expected one of {ROPE_SCALINGS}")
    if dim % 2 != 0:
        raise ValueError(f"RoPE needs an even dimension to form pairs, got {dim}")

    exponents = torch.arange(0, dim, 2).float() / dim
    inv_freq = 1.0 / (base ** exponents)
    mscale = 1.0
    if method == "pi":
        inv_freq = inv_freq / factor
    elif method == "ntk":
        stretched_base = base * factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (stretched_base ** exponents)
    else:  # yarn
        wavelength = 2 * math.pi / inv_freq
        turns = original_max_len / wavelength  # how often each hand goes round in training
        keep = ((turns - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0)  # 1 = keep speed, 0 = interpolate
        inv_freq = (1 - keep) * inv_freq / factor + keep * inv_freq
        mscale = 0.1 * math.log(factor) + 1.0

    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs) * mscale, torch.sin(freqs) * mscale


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
# THE MINIMALIST: ALiBi (Attention with Linear Biases, Press et al. 2022)
# =========================================================================
def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    How hard each head punishes distance.

    ALiBi doesn't touch the word vectors at all. It just subtracts
    `slope * distance` from every attention score. Each head gets its own slope:
    steep heads only look at close neighbours, gentle heads can look far back.
    With 8 heads the slopes are 1/2, 1/4, ..., 1/256 (a geometric sequence).

    Exactly the formula from the paper's official code, including its rule for
    head counts that are not a power of two.
    """
    def power_of_2_slopes(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = power_of_2_slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = power_of_2_slopes(closest) + get_alibi_slopes(2 * closest).tolist()[0::2][: n_heads - closest]
    return torch.tensor(slopes, dtype=torch.float)


def get_alibi_bias(n_heads: int, seq_len: int, device=None) -> torch.Tensor:
    """
    The penalty added to the attention scores, shaped [n_heads, seq_len, seq_len].

    bias[h, i, j] = -slope[h] * |i - j|  -> 0 on the diagonal, more negative further away.
    It only depends on the distance, so no table is needed and any length works.
    """
    pos = torch.arange(seq_len, device=device)
    distance = (pos[:, None] - pos[None, :]).abs().float()
    return -get_alibi_slopes(n_heads).to(device)[:, None, None] * distance


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
        
    print("task complete. the score changed with distance only, while 'apple' kept its length! (>_o)")
    print("plot twist: in 2D there is just ONE clock hand, so keep rolling and the score swings back up.")
    print("real models spin many hands at different speeds; averaged together, *that* is what makes")
    print("far-away words matter less. run `python experiment.py` to see the real decay.\n")