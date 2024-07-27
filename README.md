# AI312-Language-Modelling 🏗💾

## Overview 🌐
In the field of natural language processing (NLP), a **language model** is a statistical model that aims to capture the probability distribution of a sequence of words or tokens based on certain given datesets . **A bi-gram language model** is a specific type of n-gram language model, where n=2. It considers the probability of a word given the previous word. In a bi-gram language model, the probability of a word is only conditioned on the previous word, so the formula will be :

P(w1, w2, ..., wn) = P(w1) * P(w2|w1) * P(w3|w2) * ... * P(wn|wn-1)

## Tasks  📝
- Task-1: calculate the bi-gram probabilities of all the words/tokens in the BeRP dataset
- Task-2: calculate the probabilities of the sentences:
  - P(“show me all the Arabic food restaurants”) ?
  - P(“I am learning mathematics”) ?

## Outputs 📷
![image](https://github.com/user-attachments/assets/9f67b510-e493-4dac-b0bc-b6049f543b80)

## How to Run ⚙️
1. Download the repository 
```
 git clone https://github.com/SalwaSh/AI312-Language-Modelling.git
```
2. To run main.py
```
python main.py
```
## Contributor ✍️
- Salwa Shamma
