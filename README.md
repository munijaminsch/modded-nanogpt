# Modded-NanoGPT

This repository hosts the *NanoGPT speedrun*, in which we (collaboratively|competitively) search for the fastest algorithm to use 8 NVIDIA H100 GPUs to train a language model that attains 3.28 cross-entropy loss on the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) validation set.

(Note: Besides the main track, there is also an [optimization track](records/track_3_optimization) where we try to minimize steps subject to fixed arch/data/bsz and with unlimited wallclock budget.)

The target (3.28 validation loss on FineWeb) follows Andrej Karpathy's [GPT-2 replication in llm.c, which attains that loss after running for 45 minutes](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).
The speedrun code also descends from llm.c's [PyTorch trainer](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py), which itself descends from NanoGPT, hence the name of the repo.
Thanks to the efforts of many contributors, this repo now contains a training algorithm which attains the target performance in:
* Under 90 seconds on 8xH100 (the llm.c GPT-2 replication needed 45 minutes)
* under 400M tokens (the llm.c GPT-2 replication needed 10B)

This improvement in training speed has been brought about by the following techniques:
* Modernized architecture: Rotary embeddings, QK-Norm, and ReLU²
* The Muon optimizer [[writeup](https://kellerjordan.github.io/posts/muon/)] [[repo](https://github.com/KellerJordan/Muon)]
* Use FP8 matmul for head, and asymmetric rescale and softcap logits
* Initialization of projections to zero (muP-like)
* Skip connections from embedding to every block as well as from block 3 to 6
* Extra embeddings which are mixed into the values in attention layers (inspired by Zhou et al. 2024)
* Flash Attention 3 with long-short sliding window attention pattern (inspired by Gemma 2) and window size warmup with YaRN
* Align training batch starts with EoS and set a max document length
* Accumulate gradients for 2 steps for embedding and lm_head before updating parameters
* Backout, with single activation input for last 3 attention layers
* Polar Express implementation in Muon
* Smear module to enable 1 token look back
* Sparse attention gate
* NorMuon
* Cautious Weight Decay w/ schedule tied to LR
* Exponential decay of residual stream
* Batch size schedule
* Max seq length schedule
* Partial Key Offset
* Multi token prediction
* Untie embed and lm_head at 2/3 of training
* Additional gating on value embeddings and skip connection
* Paired head attention
* Bigram hash embedding

As well as many systems optimizations.

> **Personal note:** I forked this repo to study the training techniques listed above, particularly Muon and the architectural changes. My notes and experiments are in the `experiments/` directory (not yet pushed). If you're also learning from this codebase, the best entry point is `train_gpt2.py` — it's surprisingly readable given how much is packed in.

Contributors list (growing with each new record): [@bozavlado](https://x.com/bozavlado); [@brendanh0gan](https://x.com/brendanh0gan);
[@fernbear.bsky.social](https://bsky.app/profile/fernbear.bsky.social); [@Grad62304977](https://x.com/Grad62304977); 
[@jxbz](https://x.com/jxbz); [@kellerjordan0](https://x.com/keller
