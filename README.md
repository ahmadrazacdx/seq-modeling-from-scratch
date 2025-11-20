# Sequence Modeling from Scratch

> *What I cannot create, I do not understand.* — Richard Feynman

**Building RNNs, LSTMs, and GRUs from the ground up in NumPy. 100% Manual Calculus. 0% Autodiff.**


## Abstract
Modern deep learning frameworks (PyTorch, TensorFlow) abstract away the underlying calculus of sequence modeling, often treating architectures as black boxes. This work presents a rigorous, from-scratch implementation of fundamental Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Gated Recurrent Units (GRUs) using only **NumPy**. I explicitly derive and implement the Backpropagation Through Time (BPTT) algorithms for each architecture, verifying the gradient flow against foundational papers *(Elman, 1990; Hochreiter & Schmidhuber, 1997; Cho et al., 2014)*. This repository serves as a comprehensive educational curriculum for understanding the mathematical internals of sequence modeling, from discrete embedding lookups to the specific gating mechanisms that mitigate the vanishing gradient problem.
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

* **[01_RNN_NumPy.ipynb](./char_level_lm/01_RNN_NumPy.ipynb):** Vanilla RNN. Implements the basic recurrence relation `h_t = tanh(Wx + Uh)` and visualizes the vanishing gradient problem in practice.
* **[02_LSTM_NumPy.ipynb](./char_level_lm/02_LSTM_NumPy.ipynb):** Vanilla LSTM. Constructs the complete four-gate architecture (Forget, Input, Candidate, Output) and the cell state highway that preserves long-term gradients.
* **[03_GRU_NumPy.ipynb](./char_level_lm/03_GRU_NumPy.ipynb):** Vanilla GRU. Implements the original Gated Recurrent Unit formulation as defined by **Cho et al. (2014)**.

### Phase 2: Word-Level Modeling & Embeddings
*Objective: Transition from discrete characters to continuous dense vector representations.*

* **[01_RNN_NumPy.ipynb](./word_level_lm/01_RNN_NumPy.ipynb):** Embedding Layers. Replaces one-hot encoding with lookup tables and implements **sparse gradient updates** manually during backpropagation.
* **[02_LSTM_NumPy.ipynb](./word_level_lm/02_LSTM_NumPy.ipynb):** Word-Level LSTM. Integrates learned embeddings with the LSTM architecture to handle larger vocabularies.
* **[03_GRU_NumPy.ipynb](./word_level_lm/03_GRU_NumPy.ipynb):** Production Variants. Implements the **PyTorch definition** of the GRU (where the reset gate is applied after matrix multiplication), contrasting it with the academic paper definition.

### Phase 3: Sequence-to-Sequence Architectures *(Coming Soon)*
* **01_Encoder_Decoder_GRU_NumPy.ipynb:** Encoder-Decoder. Connects two RNNs to map input sequences to output sequences, establishing the foundation for machine translation.

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
*From `word_level_lm/02_LSTM_NumPy.ipynb` after 5000 iterations:*

```text
Iter 5000 | Loss: 1.9442
Sample: "once upon a time , on a very hot day , a thirsty crow was flying..."
```

## References

* **[RNN]** Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179-211. [[Link](https://crl.ucsd.edu/~elman/Papers/fsit.pdf)]
* **[LSTM]** Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. [[Link](https://www.bioinf.jku.at/publications/older/2604.pdf)]
* **[GRU]** Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. [[Link](https://arxiv.org/abs/1406.1078)]
* **[Adam]** Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*. [[Link](https://arxiv.org/abs/1412.6980)]

## Feedback
This repository is a living curriculum. I built this to truly understand the recurrent nets and their dynamics. If you spot a mathematical inconsistency or have a question about a derivation/code, please open an issue.

---
**Find this resource helpful? A star ⭐ is the best way to support the project!**
