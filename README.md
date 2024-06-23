# Article-Summarizer-Using-AI

An AI-based web application that provides concise summaries of articles using advanced natural language processing (NLP) techniques.

## Table of Contents

- [Introduction](#introduction)
- [Data Exploration](#data-exploration)
- [Model Selection](#model-selection)
- [Model Fine-Tuning](#model-fine-tuning)
- [Extractive Summarization](#extractive-summarization)
- [Web Application Development](#web-application-development)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

**Article-Summarizer-Using-AI** is a web application designed to summarize lengthy articles using NLP. The application allows users to upload their own articles or use sample data to generate summaries in various styles, utilizing a generative AI model.

## Data Exploration

### Dataset

The dataset used for training and evaluation is the [PubMed Summarization dataset](https://huggingface.co/datasets/ccdv/pubmed-summarization). It includes articles from PubMed with corresponding abstracts used as summaries.

1. **Loading the Dataset**:

    ```python
    from datasets import load_dataset

    pubmed_data = load_dataset("ccdv/pubmed-summarization", split='train[:1000]')
    ```

2. **Initial Data Cleaning**:

    - Remove rows with missing values to ensure data quality.

    ```python
    pubmed_data = pubmed_data.filter(lambda x: x['article'] is not None and x['abstract'] is not None)
    ```

3. **Exploratory Data Analysis**:

    - Examine the distribution of article lengths and summary lengths.
    - Identify common topics and terminology within the dataset.

    ```python
    print(pubmed_data[0])  # View the first data entry
    ```

## Model Selection

### Preprocessing

1. **Text Tokenization**:

    - Split text into sentences and words for detailed analysis.

    ```python
    from nltk.tokenize import sent_tokenize, word_tokenize

    sentences = sent_tokenize(article_text)
    words = word_tokenize(sentence)
    ```

2. **Stop Words Removal**:

    - Remove common English words that do not contribute to the summary.

    ```python
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    ```

3. **Lemmatization**:

    - Convert words to their base forms.

    ```python
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    ```

### Generative Model

1. **API Configuration**:

    - Use the `google.generativeai` library for model generation.

    ```python
    import google.generativeai as genai
    import os

    api_key = os.environ.get('your_api_key')
    genai.configure(api_key=api_key)
    ```

2. **Model Initialization**:

    - Set up the generative AI model.

    ```python
    model = genai.GenerativeModel()
    ```

## Model Fine-Tuning

### Training

- Fine-tune the model with the PubMed dataset to improve summary quality.

    ```python
    # Example pseudo-code for fine-tuning
    model.train(dataset=pubmed_data, epochs=10, learning_rate=0.001)
    ```
## Extractive Summarization

### Approach

For extractive summarization, the application uses traditional NLP techniques to identify key sentences from the article without relying on a generative model.

1. **Extractive Summary Script**:

    Rename the provided `extractive_summary.py` to `app.py` and move it to the project root:

    ```bash
    mv /mnt/data/extractive_summary.py app.py
    ```

2. **Core Logic**:

    - The extractive summarization script uses statistical and heuristic methods to identify the most important sentences in the text.

    ```python
    # Example of extractive summarization
    def extractive_summary(text):
        # Tokenize the text and rank sentences
        sentences = sent_tokenize(text)
        # Rank and select key sentences (pseudo-code)
        summary = ' '.join(sentences[:3])  # Example: Select first 3 sentences
        return summary
    ```

3. **Integration**:

    - Integrate the extractive summarization logic with the Flask application to allow users to choose between generative and extractive summaries.

    ```python
    @app.route('/summarize', methods=['POST'])
    def summarize():
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            article_text = file.read().decode("utf-8")
        else:
            sample_index = int(request.form['sample'])
            article_text = pubmed_data[sample_index]['article']

        style = request.form.get('style', 'brief')
        summary_method = request.form.get('method', 'generative')
        
        if summary_method == 'generative':
            summary_text = preprocess_and_summarize(article_text, style)
        else:
            summary_text = extractive_summary(article_text)

        return render_template('result.html', original=article_text, summary=summary_text)
    ```


### Evaluation

- Evaluate the model's performance using metrics such as ROUGE or BLEU.

    ```python
    from nltk.translate.bleu_score import sentence_bleu

    reference = [reference_summary.split()]
    candidate = generated_summary.split()
    score = sentence_bleu(reference, candidate)
    print(f'BLEU Score: {score}')
    ```

## Web Application Development

### Backend

- **Flask Setup**:

    - Initialize the Flask app and configure the login manager.

    ```python
    from flask import Flask
    from flask_login import LoginManager

    app = Flask(__name__)
    app.secret_key = 'your_secret_key'
    login_manager = LoginManager(app)
    ```

- **Routes and Authentication**:

    - Implement routes for login, registration, summarization, and logout.

    ```python
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        # login logic here
        return render_template('login.html')
    ```

### Frontend

- **Templates**:

    - Create HTML templates for the user interface.

    ```html
    <!-- templates/index.html -->
    <form action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <button type="submit">Summarize</button>
    </form>
    ```

- **User Experience**:

    - Ensure a user-friendly interface with clear instructions and feedback.

## Installation

### Prerequisites

- Python 3.7+
- Flask
- NLTK
- Generative AI Library (e.g., google.generativeai)
- An API key for generative AI

### Steps

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/Article-Summarizer-Using-AI.git
    ```

2. **Navigate to the Project Directory**:

    ```bash
    cd Article-Summarizer-Using-AI
    ```

3. **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

5. **Set Environment Variables**:

    - Create a `.env` file with your API key.

    ```plaintext
    your_api_key=<YOUR_GENERATIVE_AI_API_KEY>
    ```

6. **Download NLTK Data**:

    The script handles downloading necessary NLTK data.

## Usage

1. **Run the Application**:

    ```bash
    flask run --port=5001
    ```

2. **Access the App**:

    - Visit `http://127.0.0.1:5001` in your browser.

3. **Login/Register**:

    - Register a new account or log in with existing credentials.

4. **Summarize Articles**:

    - Upload a text file or choose a sample to summarize.

5. **View Summary**:

    - The summarized text is displayed on the results page.


Thank you for using **Article-Summarizer-Using-AI**! We hope you find it useful for your summarization needs.
