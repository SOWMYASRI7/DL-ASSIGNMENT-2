# DL-ASSIGNMENT-2




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
