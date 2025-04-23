# scratch-nmt-transformer
Custom-built Transformer for English–Chinese translation. No pretrained models, just stubbornness and PyTorch.

# Scratch NMT Transformer

Custom-built Transformer for English–Chinese translation.  
No pretrained models, just stubbornness and PyTorch.

This project is a full implementation of a Transformer-based Neural Machine Translation (NMT) system, built entirely from scratch — no shortcuts, no high-level libraries like `transformers`, just the raw PyTorch APIs and the original paper as a guide.

## Features

- Full encoder-decoder architecture  
- Positional encoding, multi-head attention, masked decoder  
- Training and inference pipeline  
- Works on small custom parallel corpora (e.g. IWSLT-style)

## Why?

Because doing things the hard way is the fastest way to understand them.  
Also because I enjoy debugging things that almost work.

## Disclaimer

It's not SOTA, it's not optimized, and it might hallucinate dragons.  
But it runs, and it taught me more than any tutorial ever could.
