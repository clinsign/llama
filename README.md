
```

2024-05-01

created acc with huggingface.co

this is the GitHub for ai models etc

python -m pip install huggingface_hub
huggingface-cli login

************


pip install --pre torch torchvision torchaudio \
 --extra-index-url https://download.pytorch.org/whl/nightly/cpu

export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0
python -c 'import torch; print(torch.cuda.is_available());'

pip install pandas

these lib are from huggingface
pip install tokenizers
pip install transformers
pip install sentencepiece


GOAL:

CLONE THE FOLLOWING REPO
RUN THE TRAINED MODEL LOCALLY AND RUN A PROMPT AGAINST IT
MODEL IS LLAMMA VERSION 1
the original model are not available on hf
this is the alternative
baffo32/decapoda-research-llama-7B-hf
https://huggingface.co/baffo32/decapoda-research-llama-7B-hf/tree/main
cloned - looks great
https://github.com/clinsign/llama

RESULT:
TRIED ON PIXEL - RUNNING THE MODEL CRASHED - NOT ENOUGH RAM/CPU
TRIED ON MAC INTEL - WORKED




install git lfs etc - only if putting a repo on huggingface
https://huggingface.co/docs/hub/en/repositories-getting-started


pip install --upgrade transformers==4.33.1
because the latest was giving errors


pip install accelerate


looks like this model was created with CUDA enabled
and cannot run on mac intel with amd radeon

AssertionError: Torch not compiled with CUDA enabled




https://www.reddit.com/r/LocalLLaMA/comments/16pxkhe/how_long_is_it_taking_you_guys_to_run_a_7b_llama/



test.py

import torch

print(torch.__version__)
print(torch.cuda.is_available())

#check for gpu
if torch.backends.mps.is_available():
   mps_device = torch.device("mps")
   x = torch.ones(1, device=mps_device)
   print (x)
else:
   print ("MPS device not found.")

inference_example.py

import llama

MODEL = 'baffo32/decapoda-research-llama-7B-hf'
# MODEL = 'decapoda-research/llama-13b-hf'
# MODEL = 'decapoda-research/llama-30b-hf'
# MODEL = 'decapoda-research/llama-65b-hf'

tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True)
#model.to('cuda')
model.to('cpu')

# here cuda was replaced with cpu

batch = tokenizer("Yo mama", return_tensors = "pt")
#print(tokenizer.decode(model.generate(batch["input_ids"].cuda(), max_length=100)[0]))

print(tokenizer.decode(model.generate(batch["input_ids"].cpu(), max_length=100)[0]))

# Expected output: "Yo mama is so fat, she has to buy two seats on the plane"

took about 6 hours and massive cpu/mem on my mac intel

(.venv) kriss-imac:llama kris$ python inference_example.py 
The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LlamaTokenizer'. 
The class this function is called from is 'LLaMATokenizer'.
/Users/kris/Documents/huggingface/.venv/lib/python3.11/site-packages/transformers/utils/generic.py:311: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/Users/kris/Documents/huggingface/.venv/lib/python3.11/site-packages/transformers/utils/generic.py:311: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████| 33/33 [00:21<00:00,  1.52it/s]
/Users/kris/Documents/huggingface/.venv/lib/python3.11/site-packages/transformers/generation/utils.py:1417: UserWarning: You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use a generation configuration file (see https://huggingface.co/docs/transformers/main_classes/text_generation )
  warnings.warn(
Yo mama is so fat, she has to buy two seats on the plane.
Yo mama is so fat, she has to buy two seats on the plane. Yo mama is so fat, she has to buy two seats on the plane. Yo mama is so fat, she has to buy two seats on the plane. Yo mama is so fat, she has to buy two seats on the plane. Yo mama is so fat,





```




_____

> **Credits:** Large parts of the code are based on the [PR](https://github.com/huggingface/transformers/pull/21955) by [Jason Phang](https://github.com/zphang). 
> Thank you for your hard work!
_____

# LLaMA. Simple. Using HuggingFace.


### What is this all about?

- Do you also want a "private GPT-3" at home?
- It also annoys you that people on the internet are excited about "llama weights" and yet there is no interface or any guide for how to use them?
- You also sick of dealing with all kinds of people on the Internet who play around with tensors then upload a code that no one can really use?

**I prepared a single repo for you with EVERYTHING you need to run LLaMA.**

Here is Everything you need for running (and training!) LLaMA using Hugging Face interface 👌

### TL;DR:

```python
tokenizer = llama.LLaMATokenizer.from_pretrained('decapoda-research/llama-7b-hf')
model = llama.LLaMAForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')
print(tokenizer.decode(model.generate(tokenizer('Yo mama', return_tensors = "pt")["input_ids"])[0]))
```

Yeah. No overengineering bullshit.

> Also: No need to clone a huge custom `transformers` repo that you later on stuck with maintaining and updating yourself. 


# What is LLaMA?

**TL;DR:** GPT model by [meta](https://ai.facebook.com/research/publications/llama-open-and-efficient-foundation-language-models/) that surpasses GPT-3, released to selected researchers but [leaked to the public](https://analyticsindiamag.com/metas-llama-leaked-to-the-public-thanks-to-4chan/).

LLaMA is a large language model [trained by Meta AI](https://ai.facebook.com/research/publications/llama-open-and-efficient-foundation-language-models/) that surpasses GPT-3 in terms of accuracy and efficiency while being 10 times smaller.

> **Paper Abstract:**
>
> We introduce LLaMA, a collection of founda- tion language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla- 70B and PaLM-540B. We release all our models to the research community.
>   

# How can I use LLaMA?

## Installation

```bash
git clone https://github.com/ypeleg/llama
```

## Usage

### 1. Import the library and choose model size

```python
import llama
MODEL = 'decapoda-research/llama-7b-hf'
```

**We currently support the following models sizes:**
- 
- Options for `MODEL`:
    - `decapoda-research/llama-7b-hf`
    - `decapoda-research/llama-13b-hf`
    - `decapoda-research/llama-30b-hf`
    - `decapoda-research/llama-65b-hf`

**Note:** The model size is the number of parameters in the model. The larger the model, the more accurate the model is, but the slower, heavier and more expensive it is to run. 

### 2. Load the tokenizer and model

```python
tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
model = llama.LLaMAForCausalLM.from_pretrained(MODEL)
model.to('cuda')
```

### 3. Encode the prompt

> For example, we will use the prompt: "Yo mama"
>   
> We will use the `tokenizer` to encode the prompt into a tensor of integers.

```python
PROMPT = 'Yo mama'
encoded = tokenizer(PROMPT, return_tensors = "pt")
```

### 4. Generate the output

> We will use the `model` to generate the output.

```python
generated = model.generate(encoded["input_ids"].cuda())[0])
``` 

### 5. Decode the output
```python
decoded = tokenizer.decode(generated)
```

### 6. Print the output

```python
print(decoded)
```

**Expected output:** "Yo mama is so fat, she has to buy two seats on the plane."
