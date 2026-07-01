# LSTM & GRU Next Word Prediction

A deep learning project that predicts the next word in a sequence using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) neural networks. This project demonstrates text generation and sequence prediction capabilities using Shakespeare's "Hamlet" as the training dataset.

## 📋 Project Description

This project aims to develop a sophisticated deep learning model for predicting the next word in a given sequence of words. By leveraging LSTM and GRU networks—two of the most effective architectures for processing sequential data—the model can learn patterns in Shakespearean text and generate contextually relevant word predictions.

### Key Features
- **Dual Architecture Support**: Both LSTM and GRU models for comparison and flexibility
- **Early Stopping**: Prevents overfitting during training with validation monitoring
- **Interactive Web Interface**: Streamlit-based application for real-time predictions
- **Pre-trained Models**: Ready-to-use trained models included in the repository
- **Comprehensive Dataset**: Trained on Shakespeare's "Hamlet" (4,818 unique words)

## 🎯 Project Workflow

### 1. **Data Collection**
The project uses the complete text of Shakespeare's "Hamlet", a rich and complex literary work that provides diverse vocabulary and sentence structures for training.

### 2. **Data Preprocessing**
- Convert all text to lowercase for uniform processing
- Tokenize text into words using TensorFlow's `Tokenizer`
- Create n-gram sequences from the text
- Pad sequences to ensure uniform input lengths
- Split data into training and testing sets

### 3. **Model Architecture**
The neural network is built with the following layers:
- **Embedding Layer**: Converts word indices to dense vector representations
- **LSTM/GRU Layers**: Two stacked recurrent layers to capture long-range dependencies
- **Dense Output Layer**: Softmax activation function for probability distribution over vocabulary
- **Total Vocabulary**: 4,818 unique words from the dataset

### 4. **Model Training**
- Uses categorical cross-entropy loss function
- Adam optimizer for efficient gradient descent
- Early stopping monitors validation loss to prevent overfitting
- Training configured with batch processing for efficiency

### 5. **Model Evaluation**
The model is tested on various example sentences to verify prediction accuracy and quality.

### 6. **Deployment**
A Streamlit web application provides an interactive interface where users can:
- Input a sequence of words
- Receive real-time predictions for the next word
- Explore the model's behavior across different inputs

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ssampathkumar104/LSTM_GRU_next_word.git
cd LSTM_GRU_next_word
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries
- `tensorflow` - Deep learning framework for building and training models
- `numpy` - Numerical computing library
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning utilities
- `streamlit` - Interactive web application framework
- `nltk` - Natural Language Toolkit for text processing
- `matplotlib` - Data visualization
- `tensorboard` - Training visualization

## 📊 Usage Examples

### Using the Streamlit Web Application

1. **Start the application**
```bash
streamlit run app.py
```

2. **Interact with the model**
   - Open your browser to `http://localhost:8501`
   - Enter a sequence of words in the text input field
   - Click "Predict Next Word" to get the prediction

**Example Inputs and Outputs:**
```
Input:  "To be or not to"
Output: "be"

Input:  "The king said"
Output: "i"

Input:  "what a piece of"
Output: "work"
```

### Using the Model Programmatically

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Predict the next word given a sequence of words."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Example usage
max_seq_length = model.input_shape[1] + 1
input_text = "To be or not to"
next_word = predict_next_word(model, tokenizer, input_text, max_seq_length)
print(f"Input: '{input_text}'")
print(f"Predicted next word: '{next_word}'")
```

### Training Your Own Model

Run the Jupyter notebooks included in the repository:

```bash
# For LSTM training
jupyter notebook training.ipynb

# For GRU training
jupyter notebook GRU_training.ipynb
```

**Training Steps in the Notebook:**
1. Load and preprocess the Hamlet text
2. Create tokenizer and convert text to sequences
3. Build the neural network architecture
4. Train with early stopping
5. Evaluate model performance
6. Save the trained model and tokenizer

## 📁 Project Structure

```
LSTM_GRU_next_word/
├── training.ipynb              # LSTM training notebook
├── GRU_training.ipynb          # GRU training notebook
├── app.py                       # Streamlit web application
├── next_word_lstm.h5           # Pre-trained LSTM model
├── next_word_gru.h5            # Pre-trained GRU model
├── tokenizer.pickle            # Tokenizer for LSTM model
├── gru_tokenizer.pickle        # Tokenizer for GRU model
├── hamlet.txt                  # Training dataset
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── Improve.docx               # Improvement suggestions document
```

## 🔧 Model Details

### LSTM Model
- **Input Shape**: Sequential word indices
- **Embedding Dimension**: Learned representations
- **LSTM Units**: 150 (first layer) + 100 (second layer)
- **Output**: Probability distribution over vocabulary
- **Model File**: `next_word_lstm.h5` (14.6 MB)
- **Tokenizer File**: `tokenizer.pickle` (187 KB)

### GRU Model
- Similar architecture using GRU cells instead of LSTM
- **Model File**: `next_word_gru.h5` (13.2 MB)
- **Tokenizer File**: `gru_tokenizer.pickle` (187 KB)

**LSTM vs GRU:**
- LSTM: More parameters, better for very long sequences
- GRU: Simpler, faster training, excellent for most text prediction tasks

## 📈 Training Configuration

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Batch Size**: 64
- **Epochs**: Variable with early stopping
- **Validation Split**: 20%
- **Early Stopping Patience**: Stops if validation loss doesn't improve for N epochs

## 💡 Key Insights

1. **Vocabulary Size**: The Hamlet text contains 4,818 unique words, providing a diverse training vocabulary
2. **Sequence Learning**: The model learns contextual patterns specific to Shakespearean English
3. **Early Stopping**: Effectively prevents overfitting by monitoring validation loss
4. **Practical Application**: Can be adapted for other text corpuses (books, code, documentation)

## 🎓 Learning Outcomes

This project demonstrates:
- Text preprocessing and tokenization techniques
- LSTM and GRU architecture implementation
- Sequence modeling for NLP tasks
- Model training with early stopping strategies
- Deployment of ML models with Streamlit
- Saving and loading trained models

## 📝 Future Improvements

Potential enhancements to the project:
- Support for multiple next-word suggestions with probabilities
- Beam search for generating longer sequences
- Fine-tuning on different text corpora
- Attention mechanisms for better context understanding
- Model quantization for faster inference
- GPU acceleration for training large models

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Share alternative implementations

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💼 Author

**S. Sampath Kumar**

## 🙏 Acknowledgments

- Shakespeare's "Hamlet" text from NLTK Gutenberg corpus
- TensorFlow and Keras for deep learning framework
- Streamlit for interactive web application framework
- scikit-learn for preprocessing utilities

---

**Last Updated**: 2024

For questions or feedback, please open an issue on the GitHub repository.
