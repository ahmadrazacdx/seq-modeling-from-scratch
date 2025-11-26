# Sequence Modeling from Scratch [![DOI](https://zenodo.org/badge/1100355653.svg)](https://doi.org/10.5281/zenodo.17720229)

> *What I cannot create, I do not understand.* — **Richard Feynman**

## Abstract
In an era where deep learning frameworks increasingly abstract the underlying mathematics of neural networks, the intuitive grasp of gradient dynamics is often lost. **Designed as a comprehensive educational resource**, this repository presents a rigorous, first-principles reconstruction of sequence modeling architectures spanning Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), Gated Recurrent Units (GRUs), and Encoder-Decoder (Seq2Seq) with Attention, implemented entirely in **NumPy**. Unlike standard tutorials that rely on automatic differentiation engines (e.g., Autograd), this curriculum explicitly derives and implements the Backpropagation Through Time (BPTT). By manually engineering the forward and backward passes, the work exposes the specific algebraic mechanisms that govern memory retention, gradient flow, and attention scoring. The implementations are validated against foundational literature *(Elman, 1990; Hochreiter & Schmidhuber, 1997; Cho et al., 2014; Bahdanau et al., 2014; Luong et al., 2015)*, providing a comprehensive, transparent view into understanding of the algorithms often hidden behind black-box APIs.
## Key Learning Outcomes
By completing this curriculum, you will be able to:

* **Master Backpropagation Through Time (BPTT):** Go beyond "black box" APIs by manually deriving and implementing the calculus behind RNNs, LSTMs, and GRUs.
* **Bridge Theory and Implementation:** Develop the skill to translate complex mathematical equations from research papers (like *Cho et al. 2014*) directly into working, vectorized NumPy code.
* **Understand NLP Efficiency:** Learn how discrete tokens are mapped to continuous vectors and how to implement sparse gradient updates for embedding layers manually.
* **Analyze Model Internals:** Gain deep intuition into *why* architectures behave the way they do, enabling you to debug convergence issues like vanishing gradients.
## Curriculum
The repository is organized into phases, guiding you from character-level to word-level progression.

### Phase 1: Character-Level Fundamentals
*Objective: Derive and implement the core algorithms of recurrence and memory.*

* **[01_RNN_NumPy.ipynb](./char_level_lm/01_RNN_NumPy.ipynb):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmadrazacdx/seq-modeling-from-scratch/blob/main/char_level_lm/01_RNN_NumPy.ipynb) Vanilla RNN. Implements the basic recurrence relation `h_t = tanh(Wx + Uh)` and visualizes the vanishing gradient problem in practice.
* **[02_LSTM_NumPy.ipynb](./char_level_lm/02_LSTM_NumPy.ipynb):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmadrazacdx/seq-modeling-from-scratch/blob/main/char_level_lm/02_LSTM_NumPy.ipynb) Vanilla LSTM. Constructs the complete four-gate architecture (Forget, Input, Candidate, Output) and the cell state highway that preserves long-term gradients.
* **[03_GRU_NumPy.ipynb](./char_level_lm/03_GRU_NumPy.ipynb):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmadrazacdx/seq-modeling-from-scratch/blob/main/char_level_lm/03_GRU_NumPy.ipynb) Vanilla GRU. Implements the original Gated Recurrent Unit formulation as defined by **Cho et al. (2014)**.

### Phase 2: Word-Level Modeling & Embeddings
*Objective: Transition from discrete characters to continuous dense vector representations.*

* **[01_RNN_NumPy.ipynb](./word_level_lm/01_RNN_NumPy.ipynb):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmadrazacdx/seq-modeling-from-scratch/blob/main/word_level_lm/01_RNN_NumPy.ipynb) Embedding Layers. Replaces one-hot encoding with lookup tables and implements **sparse gradient updates** manually during backpropagation.
* **[02_LSTM_NumPy.ipynb](./word_level_lm/02_LSTM_NumPy.ipynb):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmadrazacdx/seq-modeling-from-scratch/blob/main/word_level_lm/02_LSTM_NumPy.ipynb) Word-Level LSTM. Integrates learned embeddings with the LSTM architecture to handle larger vocabularies.
* **[03_GRU_NumPy.ipynb](./word_level_lm/03_GRU_NumPy.ipynb):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmadrazacdx/seq-modeling-from-scratch/blob/main/word_level_lm/03_GRU_NumPy.ipynb) Implements the **PyTorch definition** of the GRU (where the reset gate is applied after matrix multiplication), contrasting it with the academic paper definition.

### Phase 3: Sequence-to-Sequence Architectures
*Objective: Build complex architectures for mapping variable-length sequences.*
* **[01_Encoder_Decoder_NumPy.ipynb](./seq_2_seq/01_Encoder_Decoder_NumPy.ipynb):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmadrazacdx/seq-modeling-from-scratch/blob/main/seq_2_seq/01_Encoder_Decoder_NumPy.ipynb) Implements a full Encoder-Decoder architecture with a non-linear **Bridge** layer connecting the two networks. Features **Teacher Forcing** for training and **Autoregressive Inference** for sentence prediction, deriving the full BPTT chain rule across the bridge.
* **[02_Bahdanau_Attention_NumPy.ipynb](./seq_2_seq/02_Bahdanau_Attention_NumPy.ipynb):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmadrazacdx/seq-modeling-from-scratch/blob/main/seq_2_seq/02_Bahdanau_Attention_NumPy.ipynb) Overcomes the fixed-vector bottleneck by implementing **Bahdanau (Additive) Attention**. Features a fully differentiable attention mechanism that computes dynamic context vectors ($c_t$) and visualizes the **alignment matrix** to show source-target word relationships.
* **[03_Luong_Attention_NumPy.ipynb](./seq_2_seq/03_Luong_Attention_NumPy.ipynb):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmadrazacdx/seq-modeling-from-scratch/blob/main/seq_2_seq/03_Luong_Attention_NumPy.ipynb) Upgrades the Bahdanau model to the "Update $\rightarrow$ Look $\rightarrow$ Predict" paradigm. Implements the **General** scoring function (`h_t @ Wa @ h_s`) and introduces the **Attentional Vector** ($\tilde{h}_t$) for final predictions.

##  Implementation Details
This repository enforces production-grade engineering standards to ensure model convergence and numerical stability:

1.  **Architectural Fidelity:** All models are rigorously verified against foundational research papers (e.g., *Cho et al., 2014*; *Hochreiter & Schmidhuber, 1997*) to ensure the code aligns exactly with established research.
2.  **Adaptive Optimization:** Manual implementation of the **Adam Optimizer**, incorporating first and second moment estimation with bias correction ($\hat{m}, \hat{v}$) for stable convergence.
3.  **Gradient Stabilization:** Norm-based **Gradient Clipping** applied during Backpropagation Through Time (BPTT) to mitigate the exploding gradient problem inherent in recurrent architectures.
4.  **Numerical Precision:** Softmax and Cross-Entropy implementations utilizing the **Log-Sum-Exp** trick to prevent floating-point overflow and `NaN` errors.
5.  **Vectorized Calculus:** Matrix operations are fully vectorized using NumPy broadcasting, ensuring computational efficiency by minimizing scalar Python loops.
## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ahmadrazacdx/seq-modeling-from-scratch.git
    cd seq-modeling-from-scratch
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run a notebook:**
    Open any notebook in Google Colab or Jupyter. The data loader automatically looks for `data/thirsty_crow.txt`.

## Sample Output
*From `seq2seq/01_Encoder_Decoder_NumPy.ipynb` after 5000 iterations:*

```text
Iteration 5000 | Loss: 0.4254
Input:  The crow was thirsty .
Output: then he got an idea !
```

## References

* **[RNN]** Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179-211. [PDF](https://jontalle.web.engr.illinois.edu/Public/Elman-FindingStructureinTime.90.pdf)
* **[LSTM]** Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. [PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)
* **[GRU/Encoder-Decoder]** Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. [arXiv](https://arxiv.org/abs/1406.1078)
* **[Bahdanau Attention]** Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural machine translation by jointly learning to align and translate.* International Conference on Learning Representations (ICLR). [arXiv](https://arxiv.org/abs/1409.0473)
* **[Luong Attention]** Luong, M. T., Pham, H., & Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation*. EMNLP. [arXiv](https://arxiv.org/abs/1508.04025)
* **[Adam]** Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*. [arXiv](https://arxiv.org/abs/1412.6980)

## Citation

If you use this work, please cite it using `CITATION.cff` or the following BibTeX entry:

```bibtex
@misc{ahmad2025seqmodeling,
  author = {{Ahmad Raza}},
  title = {Sequence Modeling from Scratch: Rigorous NumPy Implementations of RNNs, LSTMs, and Attention},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ahmadrazacdx/seq-modeling-from-scratch}
}
```
## Feedback
This repository is a living curriculum. I built this to truly understand the recurrent nets and their dynamics. If you spot a mathematical inconsistency or have a question about a derivation/code, please open an issue.

---
**Find this resource helpful? A star ⭐ is the best way to support the project!**
