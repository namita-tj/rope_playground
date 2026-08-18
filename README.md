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
* **`experiment.py`** — the final boss. run this to generate a matplotlib chart proving how RoPE decays attention scores over distance compared to the old methods.

---

## =>> quick start

spin it up locally in a few seconds:

```bash
# clone the repo
git clone [https://github.com/namita-tj/rope_playground.git](https://github.com/namita-tj/rope_playground.git)
cd rope_playground

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # on windows: .venv\Scripts\Activate.ps1

# install dependencies
pip install torch matplotlib pytest numpy