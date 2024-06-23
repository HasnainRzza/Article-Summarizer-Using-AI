from flask import Flask, request, render_template
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from datasets import load_dataset

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load the PubMed Summarization dataset
pubmed_data = load_dataset("ccdv/pubmed-summarization", split='train[:1000]')  # Load a small subset for example

# Remove rows with missing values
pubmed_data = pubmed_data.filter(lambda x: x['article'] is not None and x['abstract'] is not None)

# Preprocess text function
def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words and word not in string.punctuation]
        processed_sentence = ' '.join(words)
        processed_sentences.append(processed_sentence)

    processed_text = ' '.join(processed_sentences)
    return processed_text

# Function to preprocess and summarize the article using extractive summarization
def preprocess_and_summarize(text):
    processed_text = preprocess_text(text)
    parser = PlaintextParser.from_string(processed_text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")

    summary_sentences = summarizer(parser.document, 5)  # Summarize to 5 sentences
    summary = ' '.join([str(sentence) for sentence in summary_sentences])
    return summary

@app.route('/')
def home():
    return render_template('index.html', sample_data=pubmed_data)

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files and 'sample' not in request.form:
        return "No input provided"
    
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        article_text = file.read().decode("utf-8")
    else:
        sample_index = int(request.form['sample'])
        article_text = pubmed_data[sample_index]['article']

    summary_text = preprocess_and_summarize(article_text)
    return render_template('result.html', original=article_text, summary=summary_text)

if __name__ == '__main__':
    app.run(debug=True)

