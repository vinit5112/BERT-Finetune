
# BERT Fine-Tuning Project

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/vinit5112/BERT-Finetune.git
   cd BERT-Finetune
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   Install the required packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, manually install each package:
   ```bash
   pip install transformers datasets wandb seaborn matplotlib psutil tqdm
   ```
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

## Training and Evaluation Completed

The model has already been trained and evaluated. The training script has been executed, and the results are available in the `outputs/` directory. The model and its weights have been saved in the `models/` directory.

## Dependencies

- **torch**: Deep learning framework for training BERT.
- **transformers**: Provides BERT pre-trained models and fine-tuning utilities.
- **datasets**: Library for handling datasets and loading data efficiently.
- **wandb**: For experiment tracking and visualization.

## Project Structure

```plaintext
.                
├── models/              
├── config.py            
├── bert-finetune.ipynb  
├── requirements.txt     
└── README.md            
```

The `train.py` script has already been executed to train the model, and the `evaluate.py` script has been used to evaluate it.

### Example Usage

Since the training and evaluation have already been completed, you can simply load the saved models from the `models/` directory and use them for inference or further tasks.

For example:
```python
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("models/<your_model_name>")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Example inference
inputs = tokenizer("Example sentence for inference.", return_tensors="pt")
outputs = model(**inputs)
```
