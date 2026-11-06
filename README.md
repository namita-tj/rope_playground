# rope_playground: breaking down attention (0_o)

ever wondered how AI models like GPT, Llama, or Claude actually understand the *order* of words? 

i built this repo as a learning canvas to figure it out from scratch. no dense academic math jargon—just pure python, pytorch, and interactive sandboxes where you can actually watch the tensors move. whether you're an AI student or just someone trying to peek under the hood of LLMs, this project simplifies the core mechanics of Multi-Head Attention and Rotary Position Embeddings (RoPE).

---

## [?] the learning canvas (how it works in simple terms)

before diving into the code, here is the ultimate cheat sheet for how transformers actually read.

### 1. the networking event (Q, K, V)
when a word enters the model, it splits into three parts. imagine words at a crowded networking event:
* **Query (Q):** what the word is looking for. (e.g., "bank" holds up a sign saying: *"i need words about money or rivers to know what i mean."*)
* **Key (K):** what the word is advertising. (e.g., "river" holds up a sign: *"i am a body of water."*)
* **Value (V):** the actual core meaning of the word that gets passed along if a match is made.

### 2. self-attention (the matchmaker)
instead of reading left-to-right like old AI, a transformer compares every single word's **Query** against every other word's **Key** simultaneously. if the math (dot product) scores a high match, the words blend their **Values** together. this is how "bank" realizes it means a riverbank instead of a financial institution.

### 3. the "bag of words" problem
here's the catch: because attention compares everything all at once, it has zero concept of time or order. to the math, *"the dog bit the man"* and *"the man bit the dog"* look exactly the same. if we don't fix this, the AI is essentially reading a book where all the pages are ripped out and scattered on the floor.

### 4. the old fix (vaswani sinusoidal waves)
the original 2017 transformer paper fixed this by adding sine and cosine waves directly to the words. think of it like stamping a unique mathematical wave pattern (or a page number) onto every scattered page. it works, but it's hard for the model to naturally learn the *relative* distance between words.

### 5. the modern revolution (RoPE)
modern models (llama, mistral) use **Rotary Position Embeddings (RoPE)**. instead of *adding* numbers, RoPE *rotates* the word's vector in 2D space. 
* imagine every word is a clock hand. a word at position 1 ticks forward once. position 5 ticks forward five times. 
* **the magic:** if "eat" is at pos 2 and "apple" is at pos 5, they are 3 ticks apart. if they appear later at pos 102 and 105, they are *still* 3 ticks apart. when the model compares their angles, it instantly knows their relative distance. as words get further apart, their clock hands naturally fall out of phase, and their attention score smoothly decays. closer words matter more, entirely for free!

---

## >_ what's inside (play with it)

i built this to be experimented with. clone it, break it, print the shapes.

* **`positional_encodings.py`** — the actual math for generating sinusoidal waves and RoPE 2D complex rotations (run it directly for an interactive terminal demo!).
* **`attention.py`** — building custom Multi-Head Attention ($Q, K, V$ projections and causal masking) completely from scratch.
* **`sandbox/play_custom_attention.py`** — a step-by-step trace of how a sentence splits across different attention heads.
* **`sandbox/inspect_bert_heads.py`** — peek at which word each head of a real BERT model focuses on (needs `pip install transformers`).
* **`experiment.py`** — the final boss. measures RoPE's long-term decay and saves the plot to `assets/decay_plot.png`.
* **`tests/`** — 48 checks that the math is actually right (see below).

---

## =>> quick start

spin it up locally in a few seconds:

```bash
# clone the repo
git clone https://github.com/namita-tj/rope_playground.git
cd rope_playground

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # on windows: .venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# run the experiment and the tests
python experiment.py
pytest
```

---

## \_(o_o)_/ the decay experiment

the RoPE paper promises *long-term decay*: words further apart should matter less. but there's a catch i fell into in the first version of this experiment — **decay only shows up for words that are actually related**. two random, unrelated words already score about 0 at every distance, so there's nothing to decay (my old plot was just noise!).

so `experiment.py` now measures two things, averaged over 1000 random words:

* **same word:** a word compared with *itself* placed d tokens away. without position info this is always 1.0.
* **unrelated words:** two independent random words d tokens apart.

![RoPE long-term decay](assets/decay_plot.png)

| | distance 0–7 | distance 449–512 |
| :--- | ---: | ---: |
| RoPE, same word | 0.83 | 0.23 |
| RoPE, unrelated words | ~0.00 | ~0.00 |

the same word loses about 70% of its score over 500 tokens: closer copies matter more, exactly as promised. the wiggles come from the fast-spinning clock hands drifting in and out of phase.

---

## [v] is the math actually right?

`pytest` runs 48 checks, including:

* **RoPE's defining property:** the score between positions *m* and *n* only depends on *m − n* (shift both by 1000 and nothing changes).
* **rotation keeps vector length** — spinning never stretches a word.
* **matches two independent references:** the complex-number formula `x · e^(i·m·θ)` and Hugging Face Llama's `rotate_half` formula.
* **my attention == PyTorch's built-in `scaled_dot_product_attention`**, with and without RoPE and causal masking.
* **the causal mask really hides the future:** changing the last word never changes earlier outputs.
* **order matters only with position info:** with no encoding, shuffling the words just shuffles the output.
* **any sentence length works** (an early version of `attention.py` crashed unless the sentence was exactly 2048 tokens — now there's a test so it can't come back).

CI runs all of this on every push.

