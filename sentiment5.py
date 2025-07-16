# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
import io
import base64
import traceback
import uuid
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your-secret-key-here-123'  # Change this for production

# Initialize NLP tools
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# File paths
DB_FILE = "user_responses.csv"
TRAINING_DATA_FILE = "C:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\output.csv"  # Your CSV file with statement/status
os.makedirs('models', exist_ok=True)
# os.makedirs('templates', exist_ok=True) # This line is removed as per the edit hint

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = LabelEncoder()
        self.models = {
            "lr": None,
            "svm": None
        }
        self.model_accuracy = {}
        self.vader = SentimentIntensityAnalyzer()
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize or train models"""
        try:
            self.load_models()
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.train_models_with_csv_data()
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        self.models['lr'] = joblib.load('models/lr_model.pkl')
        self.models['svm'] = joblib.load('models/svm_model.pkl')
        self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        self.label_encoder = joblib.load('models/label_encoder.pkl')
    
    def save_models(self):
        """Save models to disk"""
        joblib.dump(self.models['lr'], 'models/lr_model.pkl')
        joblib.dump(self.models['svm'], 'models/svm_model.pkl')
        joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
    
    def load_training_data(self):
        """Load training data from CSV file with statement/label columns"""
        if not os.path.exists(TRAINING_DATA_FILE):
            print(f"Training data file '{TRAINING_DATA_FILE}' not found. Terminating.")
            raise SystemExit(1)
        try:
            df = pd.read_csv(TRAINING_DATA_FILE)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if 'statement' not in df.columns or 'label' not in df.columns:
                print("CSV must contain 'statement' and 'label' columns. Terminating.")
                raise SystemExit(1)
            df = df.dropna(subset=['statement', 'label'])
            df = df[df['statement'].apply(lambda x: isinstance(x, str))]
            if len(df) < 10:
                print("Insufficient training data (need at least 10 samples). Terminating.")
                raise SystemExit(1)
            return df
        except Exception as e:
            print(f"Error loading training data: {e}. Terminating.")
            raise SystemExit(1)

    def split_train_val_test(self, X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
        # First, split off the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Now split the remaining data into train and validation
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_models_with_csv_data(self):
        """Train models with data from CSV file"""
        df = self.load_training_data()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # Preprocess and filter out empty/short/whitespace-only statements
        df['cleaned_text'] = df['statement'].apply(self.preprocess_text)
        df = df[df['cleaned_text'].str.len() > 2]
        X = df['cleaned_text']
        y = self.label_encoder.fit_transform(df['label'])
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_vec = self.vectorizer.fit_transform(X)
        # Split into train, val, test
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_train_val_test(X_vec, y)
        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
        self.models = {}
        # Logistic Regression
        self.models['lr'] = LogisticRegression(max_iter=1000)
        self.models['lr'].fit(X_train, y_train)
        # SVM
        self.models['svm'] = SVC(kernel='linear', probability=True)
        self.models['svm'].fit(X_train, y_train)
        # Accuracy
        self.model_accuracy = {
            'lr_val': self.models['lr'].score(X_val, y_val),
            'svm_val': self.models['svm'].score(X_val, y_val),
            'lr_test': self.models['lr'].score(X_test, y_test),
            'svm_test': self.models['svm'].score(X_test, y_test)
        }
        self.save_models()
        print(f"Models trained successfully with {len(df)} samples")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"LR Validation Accuracy: {self.model_accuracy['lr_val']:.2f}, SVM Validation Accuracy: {self.model_accuracy['svm_val']:.2f}")
        print(f"LR Test Accuracy: {self.model_accuracy['lr_test']:.2f}, SVM Test Accuracy: {self.model_accuracy['svm_test']:.2f}")
    
    def preprocess_text(self, text):
        """Fast text cleaning for VADER: lowercase, remove punctuation/numbers, strip whitespace. Skip empty/short/whitespace-only."""
        text = str(text)
        if not text or text.isspace() or len(text.strip()) < 3:
            return ''
        _punct_num = re.compile(r'[^a-zA-Z\s]')
        text = text.lower()
        text = _punct_num.sub('', text)
        text = text.strip()
        return text
    
    def split_statements(self, text):
        """Split input into sentences for future queries"""
        try:
            blob = TextBlob(text)
            sentences_attr = getattr(blob, 'sentences', None)
            # Check if sentences_attr is iterable
            if sentences_attr is not None and hasattr(sentences_attr, '__iter__'):
                sentences = list(sentences_attr)
                if not sentences:
                    return [text]
                return [str(sentence) for sentence in sentences]
            else:
                return [text]
        except Exception as e:
            print(f"TextBlob sentence split error: {e}")
            return [text]

    def analyze_sentiment(self, text):
        """Analyze text with all available models: Logistic Regression, SVM, TextBlob, and VADER"""
        from sklearn.metrics import precision_score, recall_score, accuracy_score
        try:
            patient_id = str(uuid.uuid4())
            statements = self.split_statements(text)
            all_results = []
            for statement in statements:
                statement = str(statement)
                cleaned_text = self.preprocess_text(statement)
                features = self.vectorizer.transform([cleaned_text])

                # TextBlob sentiment on cleaned text
                try:
                    tb_blob = TextBlob(cleaned_text)
                    tb_sent = tb_blob.sentiment
                    polarity = float(getattr(tb_sent, 'polarity', 0.0))
                    subjectivity = float(getattr(tb_sent, 'subjectivity', 0.0))
                    if polarity > 0.3:
                        tb_sentiment = "Positive"
                    elif polarity < -0.3:
                        tb_sentiment = "Negative"
                    else:
                        tb_sentiment = "Neutral"
                    tb_confidence = min(0.99, max(0.51, (abs(polarity) + subjectivity)/2))
                except Exception as tb_err:
                    print(f"TextBlob error: {tb_err}")
                    polarity = 0.0
                    subjectivity = 0.0
                    tb_sentiment = "Unknown"
                    tb_confidence = 0.0

                # VADER sentiment (always included)
                try:
                    vader_scores = self.vader.polarity_scores(statement)
                    compound = vader_scores['compound']
                    if compound >= 0.05:
                        vader_sentiment = "Positive"
                    elif compound <= -0.05:
                        vader_sentiment = "Negative"
                    else:
                        vader_sentiment = "Neutral"
                except Exception as vader_err:
                    print(f"VADER error: {vader_err}")
                    vader_sentiment = "Unknown"
                    compound = 0.0

                results = {
                    "patient_id": patient_id,
                    "text": statement,
                    "cleaned_text": cleaned_text,
                    "timestamp": datetime.now().isoformat(),
                    "sentiment_textblob": tb_sentiment,
                    "confidence_textblob": tb_confidence,
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "sentiment_vader": vader_sentiment,
                    "vader_compound": compound  # Show VADER sentiment and score
                }

                # Logistic Regression
                try:
                    if self.models['lr']:
                        proba = self.models['lr'].predict_proba(features)[0]
                        pred = self.models['lr'].predict(features)[0]
                        results["sentiment_lr"] = self.label_encoder.inverse_transform([pred])[0]
                        results["confidence_lr"] = float(max(proba))
                except Exception as lr_err:
                    print(f"Logistic Regression error: {lr_err}")
                    results["sentiment_lr"] = "Unknown"
                    results["confidence_lr"] = 0.0

                # SVM
                try:
                    if self.models['svm']:
                        proba = self.models['svm'].predict_proba(features)[0]
                        pred = self.models['svm'].predict(features)[0]
                        results["sentiment_svm"] = self.label_encoder.inverse_transform([pred])[0]
                        results["confidence_svm"] = float(max(proba))
                except Exception as svm_err:
                    print(f"SVM error: {svm_err}")
                    results["sentiment_svm"] = "Unknown"
                    results["confidence_svm"] = 0.0

                self.save_to_db(results)
                all_results.append(results)

            # Calculate precision, recall, accuracy for the models on the test set
            metrics = {}
            try:
                # Load test set from training data
                df = self.load_training_data()
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                df['cleaned_text'] = df['statement'].apply(self.preprocess_text)
                X = self.vectorizer.transform(df['cleaned_text'])
                y_true = self.label_encoder.transform(df['label'])
                for model_key, model in self.models.items():
                    if model is not None:
                        y_pred = model.predict(X)
                        metrics[model_key] = {
                            'accuracy': accuracy_score(y_true, y_pred),
                            'precision': precision_score(y_true, y_pred, average='weighted', zero_division="warn"),
                            'recall': recall_score(y_true, y_pred, average='weighted', zero_division="warn")
                        }
            except Exception as metric_err:
                print(f"Error calculating metrics: {metric_err}")

            return {
                "patient_id": patient_id,
                "statements": all_results,
                "model_accuracy": self.model_accuracy,
                "metrics": metrics  # Add metrics to output
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            traceback.print_exc()
            return {
                "text": text,
                "error": str(e)
            }
    
    def save_to_db(self, results):
        """Save analysis results to CSV"""
        try:
            df = pd.DataFrame([results])
            if not os.path.exists(DB_FILE):
                df.to_csv(DB_FILE, index=False)
            else:
                df.to_csv(DB_FILE, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    def generate_visualizations(self):
        """Create visualizations of sentiment analysis history"""
        try:
            if not os.path.exists(DB_FILE) or os.path.getsize(DB_FILE) == 0:
                return None
                
            df = pd.read_csv(DB_FILE, on_bad_lines='skip')
            if len(df) < 1:
                return None
                
            plt.figure(figsize=(15, 10))
            
            # Model comparison plot
            plt.subplot(2, 2, 1)
            model_accuracies = {}
            if 'confidence_lr' in df.columns:
                model_accuracies['Logistic Regression'] = df['confidence_lr'].mean()
            if 'confidence_svm' in df.columns:
                model_accuracies['SVM'] = df['confidence_svm'].mean()
            
            if model_accuracies:
                sns.barplot(x=list(model_accuracies.keys()), 
                            y=list(model_accuracies.values()),
                            palette="viridis")
                plt.title("Average Confidence by Model")
                plt.ylabel("Confidence Score")
                plt.xticks(rotation=45)
            
            # Sentiment distribution plot
            plt.subplot(2, 2, 2)
            if 'sentiment_lr' in df.columns:
                sentiment_counts = df['sentiment_lr'].value_counts()
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
                            palette="coolwarm")
                plt.title("Sentiment Distribution")
                plt.ylabel("Count")
            
            # Confidence over time plot
            plt.subplot(2, 2, 3)
            if 'timestamp' in df.columns and 'confidence_lr' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                if 'confidence_lr' in df.columns:
                    sns.lineplot(x='timestamp', y='confidence_lr', data=df,
                                 label='Logistic Regression')
                if 'confidence_svm' in df.columns:
                    sns.lineplot(x='timestamp', y='confidence_svm', data=df,
                                 label='SVM')
                plt.title("Confidence Over Time")
                plt.ylabel("Confidence Score")
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
            plt.close()
            
            return plot_url
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            return None

analyzer = SentimentAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    results = None
    error = None
    patient_id = None
    model_accuracy = None
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if text:
            results = analyzer.analyze_sentiment(text)
            if 'error' in results:
                error = results['error']
                results = None
            else:
                plot_url = analyzer.generate_visualizations()
                patient_id = results.get('patient_id')
                model_accuracy = results.get('model_accuracy')
        else:
            error = "Please enter some text to analyze"
    return render_template('index.html', results=results, plot_url=plot_url, error=error, patient_id=patient_id, model_accuracy=model_accuracy)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({"error": "No text provided"}), 400
    
    results = analyzer.analyze_sentiment(data['text'])
    if 'error' in results:
        return jsonify({"error": results['error']}), 500
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)