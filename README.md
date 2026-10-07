# 🛡️ SMS Spam Classifier
 
A machine-learning web app that classifies text messages as **Spam** or **Ham** (legitimate), built with Python, scikit-learn and Streamlit.
 
The model is a tuned **linear Support Vector Machine** trained on **TF-IDF** features from the SMS Spam Collection dataset. The app also explains *why* a message was flagged by showing the words that influenced the decision.
 
---
 
## Features
 
| Page | What it does |
|------|--------------|
| **Classify a message** | Paste a message to get Spam/Ham, the SVM decision margin, a signal-strength label, a spam indicator bar and a chart of the most influential words. Includes one-click example messages. |
| **Batch classification** | Upload a CSV (choose the text column) or paste one message per line, then download the predictions as a CSV. |
| **Model & experiments** | Accuracy comparison of Bag-of-Words, TF-IDF and Word2Vec experiments, deployed model settings, and the strongest spam/ham words overall. |
| **About the project** | Plain-language summary of the method and its limitations. |
 
---
 
## 📂 Project structure
 
```
sms_spam_classifier/
├── app.py                                       # Streamlit application
├── requirements.txt                             # Python dependencies
├── best_svm_model.joblib                        # Trained SVM (linear, C=10)
├── tfidf_vectorizer.joblib                      # Fitted TF-IDF vectorizer
├── Combined_Spam_Classification_Project.ipynb   # Full experiment notebook (optional)
└── README.md
```
 
`app.py`, `requirements.txt` and both `.joblib` files must sit in the **same folder** (the repository root when deploying).
 
---
 
## How the model was built
 
1. **Data**: SMS Spam Collection, 5,572 labelled messages (~13% spam).
2. **Cleaning**: removed non-letters, lowercased, removed English stopwords, applied Porter stemming.
3. **Target encoding**: `True = Ham`, `False = Spam` (via `pd.get_dummies`).
4. **Split**: 80/20 train-test. Vectorizers were fitted on the training data only to avoid data leakage.
5. **Feature techniques compared**
   - Bag-of-Words (2,500 features, 1–2 grams)
   - TF-IDF (2,500 features, 1–2 grams)
   - Averaged Word2Vec (100 dimensions, trained from scratch)
6. **Classifiers compared on each technique**: Naive Bayes, Logistic Regression, Random Forest, Extra Trees, SVM.
7. **Tuning**: `GridSearchCV` for every model, comparing train vs test accuracy to watch for overfitting, plus confusion matrices.
8. **Final choice**: tuned linear **SVM on TF-IDF** (`C=10`), chosen for strong accuracy, fast inference and the ability to explain predictions through its linear weights.
9. **Saved** with `joblib` for use in the app.
### Key observations
- Bag-of-Words and TF-IDF with linear models outperformed Word2Vec trained from scratch, which needs far more data than ~5.5k short messages.
- Spam is the minority class, so **spam recall and precision** matter more than overall accuracy.
- A linear kernel lets the app show per-word contributions for every prediction.
 
---
 
## Run locally
 
```bash
# 1. (Recommended) create a clean environment
conda create -n spam-app python=3.12 -y
conda activate spam-app
 
# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Start the app
streamlit run app.py
```
 
The app opens at `http://localhost:8501`.
 
If you have several Python installations, use the same one for both steps:
 
```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```
 
---
 
## Deploy on Streamlit Community Cloud
 
1. Push these files to the **root** of a GitHub repository: `app.py`, `requirements.txt`, `best_svm_model.joblib`, `tfidf_vectorizer.joblib`.
2. On [share.streamlit.io](https://share.streamlit.io), choose **Create app** and select the repository and `app.py`.
3. Open **Advanced settings** and set **Python version to 3.12** (the model needs scikit-learn 1.6.1, which has no builds for newer Python versions).
4. Click **Deploy**.
**Troubleshooting**
- `ModuleNotFoundError: No module named 'joblib'`: the dependency file is not being read. Make sure it is named exactly `requirements.txt`, sits in the repo root, and no other dependency file (`pyproject.toml`, `Pipfile`, `environment.yml`) is present.
- Version warnings or load errors for the model: scikit-learn must be exactly `1.6.1`, the version used to save the model.
---
 
## Requirements
 
```
streamlit>=1.35
scikit-learn==1.6.1
joblib>=1.3
numpy>=1.26
pandas>=2.0
nltk>=3.8
```
 
NLTK is used only for the Porter stemmer. The English stopword list is embedded in `app.py`, so no NLTK data download is needed.
 
---
 
## Limitations
 
- Trained on English SMS messages from a specific period; newer scam styles, other languages and long emails may be misclassified.
- The spam indicator is derived from the SVM margin and is **not a calibrated probability**.
- Messages that contain only numbers, symbols or stopwords have no usable features, so results for them are unreliable.
- Use it as a decision aid, not as the only line of defence against spam.
---
 
## Tech stack
 
Python · pandas · NumPy · scikit-learn · NLTK · Gensim (Word2Vec experiments) · Streamlit · joblib

---
 
## Author
 
**Tarun Kumar** 
