from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
import os
import google.generativeai as genai
from datasets import load_dataset

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a more secure key

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Dummy user database
users = {
    'testuser': {
        'password': generate_password_hash('password123', method='pbkdf2:sha256')
    }
}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# Check if API key is set
api_key = os.environ.get('your_api_key')
if not api_key:
    print("Error: API key environment variable is not set.")
    exit(1)

# Configure genai module
genai.configure(api_key=api_key)

# Initialize generative model
model = genai.GenerativeModel()

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

# Function to preprocess and summarize the article using generative AI
def preprocess_and_summarize(text, style):
    processed_text = preprocess_text(text)
    
    # Generate content using generative AI
    prompt = f"Summarize the following text in a {style} style: {processed_text}"
    response = model.generate_content(prompt)
    
    if response and response.text:
        summary = response.text
    else:
        summary = "Could not generate summary."

    return summary

@app.route('/')
@login_required
def home():
    return render_template('index.html', sample_data=pubmed_data)

@app.route('/summarize', methods=['POST'])
@login_required
def summarize():
    if 'file' not in request.files and 'sample' not in request.form:
        return "No input provided"
    
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        article_text = file.read().decode("utf-8")
    else:
        sample_index = int(request.form['sample'])
        article_text = pubmed_data[sample_index]['article']

    style = request.form.get('style', 'brief')
    summary_text = preprocess_and_summarize(article_text, style)
    return render_template('result.html', original=article_text, summary=summary_text)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and check_password_hash(users[username]['password'], password):
            user = User(username)
            login_user(user)
            flash('Logged in successfully.')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            flash('Username already exists.')
        else:
            users[username] = {
                'password': generate_password_hash(password, method='pbkdf2:sha256')
            }
            flash('Registration successful. You can now log in.')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
