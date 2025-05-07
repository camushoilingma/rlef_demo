# Reinforcement Learning from Execution Feedback (RLEF)

This project implements **RLEF (Reinforcement Learning from Execution Feedback)** using the **CodeLlama-7B-Instruct** model with PPO fine-tuning. 
The goal is to train the model to generate Python functions that pass test cases, leveraging the concept of **execution feedback** to guide learning.

---

## Version: Simple RLEF

This version of RLEF uses two basic prompts to train a small, testable model. It is designed for quick testing and understanding of the RLEF process without the complexity of a full dataset.

### Prompts Used:
- **Prompt 1:** Write a Python function that returns the square of a number.  
- **Prompt 2:** Write a function to check if a string is a palindrome.

---

## 📄 Paper Reference

This implementation is inspired by the method described in:  

**Improving Code Generation by Training with Execution Feedback**  
**Authors:** Chiyu Zhang, Ruochen Xu, et al.  
**Paper Link:** [arXiv:2410.02089v2](https://arxiv.org/abs/2410.02089)  
