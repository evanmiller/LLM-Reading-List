Just helping myself keep track of LLM papers that I‘m reading, with an emphasis on inference and model compression.

Transformer Architectures

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - Multi-Query Attention
* [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
* [Augmenting Self-attention with Persistent Memory](https://arxiv.org/abs/1907.01470) (Meta 2019)
* [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185) (Meta 2023)
* [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)

Foundation Models

* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
* [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745)
* [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (OpenAI) - GPT-2
* [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
* [OpenLLaMA: An Open Reproduction of LLaMA](https://github.com/openlm-research/open_llama)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
* [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

Position Encoding

* [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
* [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

KV Cache

* [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) (Jun. 2023)
* [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai)
* [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

Activation

* [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
* [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

Pruning

* [Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) (1990)
* [Optimal Brain Surgeon](https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf) (1993)
* [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/abs/2208.11580) (Jan. 2023) - Introduces Optimal Brain Quantization based on the Optimal Brain Surgeon
* [Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](https://arxiv.org/abs/1705.07565)
* [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)
* [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695) - Introduces Wanda (pruning with Weights and Activations)

Quantization

* [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) - Quantization with outlier handling. Might be solving the wrong problem - see "Quantizable Transformers" below.
* [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) - Another approach to quantization with outliers
* [Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568) (Qualcomm 2020) - Introduces AdaRound
* [Understanding and Overcoming the Challenges of Efficient Transformer Quantization](https://arxiv.org/abs/2109.12948) (Qualcomm 2021)
* [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304) (Cornell Jul. 2023) - Introduces incoherence processing
* [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629) (Berkeley Jun. 2023)

Normalization

* [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
* [Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing](https://arxiv.org/abs/2306.12929) - Introduces gated attention and argues that outliers are a consequence of normalization

Sparsity and rank compression

* [Compressing Pre-trained Language Models by Decomposition](https://aclanthology.org/2020.aacl-main.88/) - vanilla SVD composition to reduce matrix sizes
* [Language model compression with weighted low-rank factorization](https://arxiv.org/abs/2207.00112) - Fisher information-weighted SVD
* [Numerical Optimizations for Weighted Low-rank Estimation on Language Model](https://arxiv.org/abs/2211.09718) - Iterative implementation for the above
* [Weighted Low-Rank Approximation](https://cdn.aaai.org/ICML/2003/ICML03-094.pdf) (2003)
* [Transformers learn through gradual rank increase](https://arxiv.org/abs/2306.07042)
* [Pixelated Butterfly: Simple and Efficient Sparse training for Neural Network Models](https://arxiv.org/abs/2112.00029)
* [Scatterbrain: Unifying Sparse and Low-rank Attention Approximation](https://arxiv.org/abs/2110.15343)
* [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/abs/2306.11222)
* [LadaBERT: Lightweight Adaptation of BERT through Hybrid Model Compression](https://arxiv.org/abs/2004.04124)
* [KroneckerBERT: Learning Kronecker Decomposition for Pre-trained Language Models via Knowledge Distillation](https://arxiv.org/abs/2109.06243)
* [TRP: Trained Rank Pruning for Efficient Deep Neural Networks](https://arxiv.org/abs/2004.14566) - Introduces energy-pruning ratio

Fine-tuning

* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
* [DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation](https://arxiv.org/abs/2210.07558) - works over a range of ranks
* [Full Parameter Fine-tuning for Large Language Models with Limited Resources](https://arxiv.org/abs/2306.09782)

Sampling

* [Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity](https://arxiv.org/abs/2007.14966)
* [Stay on topic with Classifier-Free Guidance](https://arxiv.org/abs/2306.17806)

Scaling

* [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (Google Nov. 2022) - Pipeline and tensor parallelization for inference
* [Megatron-LM](https://arxiv.org/abs/1909.08053) (Nvidia Mar. 2020) - Intra-layer parallelism for training

Mixture of Experts

* [Adaptive Mixtures of Local Experts](https://github.com/mtotolo/nnetworks_HG/blob/master/Adaptive-mixtures-of-local-experts.pdf) (1991, remastered PDF)
* [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Google 2017)
* [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) (Google 2022)

Watermarking

* [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)


More

* [Efficient Deep Learning Systems: Week 9, Compression](https://github.com/mryab/efficient-dl-systems/tree/main/week09_compression)
