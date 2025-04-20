# DL-ASSIGNMENT-2

This repository contains two deep learning projects:

1. Latin to Devanagari Transliteration using RNN-based Seq2Seq Model  
2. GPT-2 Song Lyrics Generation: Ed Sheeran & Coldplay

---

## 1. Latin to Devanagari Transliteration (Seq2Seq Model)

### Overview
This project implements a character-level sequence-to-sequence (seq2seq) model to transliterate Latin-script Hindi words into Devanagari script using an RNN-based encoder-decoder architecture.

### Model Architecture
- **Embedding Layer**: Converts input character indices into dense vectors.
- **Encoder**: Configurable RNN (SimpleRNN / LSTM / GRU) with customizable layers.
- **Decoder**: Takes encoder's final state to generate output characters one at a time.
- **Dense Layer**: Predicts output character probabilities at each decoding step.

### Model Flexibility
Customizable parameters:
- Embedding dimension (m)
- Hidden state dimension (k)
- Number of encoder and decoder layers
- RNN Cell Type: SimpleRNN, LSTM, or GRU

### Dataset
Uses the Hindi transliteration subset of the Dakshina dataset:
- `/content/hi.translit.sampled.train.tsv`
- `/content/hi.translit.sampled.dev.tsv`
- `/content/hi.translit.sampled.test.tsv`

Each file contains parallel Latin and Devanagari word pairs.

---

### (a) Total Number of Computations

**Given:**
- Embedding size (m) = 256  
- Hidden state size (k) = 256  
- Vocabulary size (V) = 60  
- Sequence length (T) = 20  

**Formula:**Total Computation = T × [8k(m + k) + kV]

**Calculation:**= 20 × [8×256(256+256) + 256×60] = 21.28 million operations

---

### (b) Total Number of Parameters 

**Formula:**Total Params = 2Vm + 8k(m + k + 1) + kV + V


**Calculation:**= 2×60×256 + 8×256(256+256+1) + 256×60 + 60 = 1.10 million parameters


---

### (c) Best Model Accuracy and Predictions

- **Best Cell Type:** LSTM  
- **Test Accuracy:** 29.12%

### Sample of Preprocessed Data
Latin Input (Encoded): [ 1 14  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
Latin Input (Decoded): an
Devanagari Target (Encoded): [6 4 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
Devanagari Target (Decoded): अं

--------------------------------------------------

Latin Input (Encoded): [ 1 14 11  7  1 14  9 20  0  0  0  0  0  0  0  0  0  0  0  0]
Latin Input (Decoded): ankganit
Devanagari Target (Encoded): [ 6  4 18 20 32 53 33  2  0  0  0  0  0  0  0  0  0  0  0  0]
Devanagari Target (Decoded): अंकगणित

--------------------------------------------------

Latin Input (Encoded): [21 14  3 12  5  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
Latin Input (Decoded): uncle
Devanagari Target (Encoded): [ 6  4 18 45  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
Devanagari Target (Decoded): अंकल

#### Sample Predictions

| Input      | Predicted |
|------------|-----------|
| ank        | अंक       |
| anka       | अंका      |
| ankit      | अंकित     |
| anakon     | अनकों     |
| ankhon     | अंखोह     |
| ankon      | अंकों     |
| angkor     | अंगोग     |
| ankor      | अंकोर     |
| angaarak   | अंगारक    |
| angarak    | अंगररक    |

#### Visualization

![Training Loss Curve](https://github.com/user-attachments/assets/5ad3f7d2-a429-4efc-b60c-6b5a641d8695)

---

## 2. GPT-2 Song Lyrics Generation: Ed Sheeran & Coldplay

### Goal
Fine-tune GPT-2 on lyrics from Ed Sheeran and Coldplay to generate stylistically similar lyrics from custom prompts.

### Technologies Used

- **Python**
- **Hugging Face Transformers**: GPT-2, Trainer, Tokenizer
- **PyTorch**
- **Pandas**
- **Excel**: Dataset stored in `EdSheeran_Coldplay_Lyrics.xlsx` (LYRICS column)
- **Other**: OS, Torch (for GPU acceleration)

### Fine-tuning Pipeline

1. Load and preprocess lyrics dataset.
2. Tokenize text with GPT-2 tokenizer.
3. Fine-tune using Hugging Face `Trainer` API.
4. Generate lyrics from custom prompts.

---

### Example Prompt

> **Prompt:** "When the lights go out"  
> **Generated:** "And the silence starts to scream, I’m holding on to a dream..."

---

