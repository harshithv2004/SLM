#  TinyStories Language Model (SLM)

A lightweight **Transformer-based Language Model** built from scratch using **PyTorch**, inspired by [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and **nanoGPT**.  
The model is trained to generate short, coherent, and creative stories 
— A project for Myself  exploring **Natural Language Processing (NLP)** and **Deep Learning** concepts in an  experimental setting.

---

##  Features

-  Implemented fully in **Python + PyTorch**
-  Uses **TinyStories dataset** via Hugging Face’s `datasets` library
-  Tokenization handled with **tiktoken (GPT-2 encoding)**
-  Transformer architecture with attention and learning rate scheduling
-  Includes checkpoint saving and easy retraining
-  Generate creative short stories from simple prompts

---

##  Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3.10+ |
| Framework | PyTorch |
| Dataset | Hugging Face `datasets` |
| Tokenizer | `tiktoken` |
| Utilities | `numpy`, `tqdm` |

---

##  Installation & Setup

Follow these steps to set up the project on your **local machine** 🖥️

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/harshithv2004/SLM
cd SLM
```

### 2️⃣ Create a Virtual Environment

**Using Conda**
```bash
conda create -n tinystory python=3.10
conda activate tinystory
```

**Using venv**
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

##  Training the Model

Open the Jupyter notebook and run all cells to train the model:

```bash
jupyter notebook tinystory_new.ipynb
```

or, in VS Code:
```bash
code tinystory_new.ipynb
```

Training will:
- Load and preprocess the **TinyStories dataset**
- Tokenize text with `tiktoken`
- Train a small Transformer model
- Save the model weights as `best_model_params.pt`

---

##  Generating Stories

After training, use the following snippet to generate new stories:

```python
import torch
import tiktoken

# Load tokenizer and model
enc = tiktoken.get_encoding("gpt2")
model = torch.load("best_model_params.pt")
model.eval()

# Generate text
sentence = "A little girl went to the woods"
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim=0)
y = model.generate(context, max_new_tokens=200)

print(enc.decode(y.squeeze().tolist()))
```

---

##  Project Structure

```
TinyStories-SLM/
│
├── tinystory_new.ipynb        # Main notebook for training & generation
├── best_model_params.pt       # Saved model weights
├── train.bin                  # Encoded dataset for training
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

##  Future Enhancements

-  Add **attention visualization** for interpretability  
-  Implement **evaluation metrics** (e.g., BLEU, perplexity)  
-  Optimize **GPU training performance**  
-  Fine-tune on other creative writing datasets  

---

## 👨‍💻 Author

**Harshith Veerapur**  
🎓 CSE Student 
📧 [Email me](mailto:harshitveerapur@example.com)  
🌐 [LinkedIn](https://linkedin.com/in/harshitveerapur) 

---

⭐ *If you like this project, consider giving it a star on GitHub!*

