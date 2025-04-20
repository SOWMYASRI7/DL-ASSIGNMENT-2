# DL-ASSIGNMENT-2

1.Latin to Devanagari Transliteration using RNN-based Seq2Seq Model

Overview:

This project implements a flexible sequence-to-sequence (seq2seq) model for transliterating Latin-script Hindi text into Devanagari script. The model uses RNN-based encoders and decoders at the character level.

The model is built with the following architecture:

Character embedding layer for input characters

Encoder RNN (configurable: RNN / LSTM / GRU)

Decoder RNN that generates one character at a time

Dense layer for output character prediction

Model Flexibility
You can customize:

Embedding dimension (m)

Hidden state dimension (k)

Number of layers in encoder and decoder

RNN Cell type: SimpleRNN, LSTM, or GRU

These parameters are passed into the model-building function to allow easy experimentation.

Dataset:
The model is trained using the Dakshina dataset, specifically the Hindi transliteration subset:

/content/hi.translit.sampled.train.tsv

/content/hi.translit.sampled.dev.tsv

/content/hi.translit.sampled.test.tsv

Each file contains parallel examples of:

Latin transliterations of Hindi words

Corresponding Devanagari representations

Architecture:
Embedding Layer: Transforms one-hot character indices to dense vectors of dimension m.

Encoder:

One or more RNN layers (configurable)

Processes the input sequence

Outputs the final hidden state(s)

Decoder:

Takes the encoder's final state as its initial state

Generates one output character at each time step


✅ Best Model and Results
🔍 Grid Sweep over Cell Types
Explored SimpleRNN, GRU, and LSTM architectures

Trained each for 10 epochs

Evaluated accuracy on test set

🏆 Best Model:
Cell Type: LSTM

Embedding Dim: 64

Hidden Dim: 128

Layers: 1 encoder, 1 decoder

Test Accuracy: XX.XX% (replace with your actual result)

🔤 Sample Predictions
Input (Latin)	Predicted (Devanagari)	Actual (Devanagari)
namaste	नमस्ते	नमस्ते
bharat	भारत	भारत
shukriya	शुक्रिया	शुक्रिया
dilli	दिल्ली	दिल्ली

📈 Visualization
Training loss curves for each model are plotted to show convergence across epochs.




  2.GPT-2 Song Lyrics Generation: Ed Sheeran & Coldplay

The model is fine-tuned on a dataset containing lyrics from these two artists, allowing it to generate creative and stylistically similar lyrics based on user prompts.

The goal of this project is to:

Fine-tune the GPT-2 model on lyrics from Ed Sheeran and Coldplay.

Use the fine-tuned model to generate new lyrics based on input prompts.

Technologies and Libraries Used
Python:

The core language for writing the code and running the scripts.

Hugging Face Transformers:

GPT-2: A pre-trained transformer model used for language generation. The model is fine-tuned on the lyrics dataset (Ed Sheeran & Coldplay) to generate song lyrics in the same style.

Trainer: A utility from the Hugging Face library to simplify the training process, which handles batching, optimization, and checkpointing.

Tokenizer: Tokenizes input text (lyrics) to convert it into a format suitable for GPT-2.

PyTorch:

Deep Learning Framework: Used for defining and training the GPT-2 model. PyTorch handles the neural network's operations, including training and inference.

Pandas:

Data Manipulation: Used to read and clean the lyrics dataset stored in Excel format. The dataset is preprocessed to ensure that the lyrics are properly formatted for training.

Excel:

Dataset Format: The lyrics dataset is stored in an Excel file (EdSheeran_Coldplay_Lyrics.xlsx). Each song's lyrics are stored in the "LYRICS" column.

Other Python Libraries:

OS: For file path handling.

Torch: To leverage GPU acceleration (if available) for faster model training.
