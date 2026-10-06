import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem.porter import PorterStemmer

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_svm_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"

st.set_page_config(page_title="Spam SMS Classifier", page_icon="🛡️", layout="wide")

# NLTK English stopword list, embedded so the app needs no download (works offline)
STOPWORDS = frozenset("""
a about above after again against ain all am an
and any are aren aren't as at be because been
before being below between both but by can couldn couldn't
d did didn didn't do does doesn doesn't doing don
don't down during each few for from further had hadn
hadn't has hasn hasn't have haven haven't having he he'd
he'll he's her here hers herself him himself his how
i i'd i'll i'm i've if in into is isn
isn't it it'd it'll it's its itself just ll m
ma me mightn mightn't more most mustn mustn't my myself
needn needn't no nor not now o of off on
once only or other our ours ourselves out over own
re s same shan shan't she she'd she'll she's should
should've shouldn shouldn't so some such t than that that'll
the their theirs them themselves then there these they they'd
they'll they're they've this those through to too under until
up ve very was wasn wasn't we we'd we'll we're
we've were weren weren't what when where which while who
whom why will with won won't wouldn wouldn't y you
you'd you'll you're you've your yours yourself yourselves
""".split())


# ---------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    # Linear-kernel SVC exposes one weight per TF-IDF feature -> explainability
    coefs = model.coef_.ravel() if hasattr(model, "coef_") else None
    return model, vectorizer, PorterStemmer(), set(STOPWORDS), coefs


model, vectorizer, stemmer, STOP, COEFS = load_assets()
FEATURES = np.array(vectorizer.get_feature_names_out())


# In training, y = (label == 'ham'), so  True/positive class = HAM, False = SPAM.
# decision_function > 0  -> Ham,  < 0 -> Spam.


# ---------------------------------------------------------- preprocessing
def preprocess(text: str) -> str:
    words = re.sub("[^a-zA-z]", " ", str(text)).lower().split()
    return " ".join(stemmer.stem(w) for w in words if w not in STOP)


def predict(texts):
    cleaned = [preprocess(t) for t in texts]
    X = vectorizer.transform(cleaned)
    X_dense = X.toarray()
    scores = model.decision_function(X_dense)
    labels = np.where(scores > 0, "Ham", "Spam")
    return cleaned, X, scores, labels


def spam_score(d):
    return 1 / (1 + np.exp(2.0 * d))


def strength(d):
    a = abs(d)
    return "Very strong" if a > 1.5 else "Strong" if a > 0.75 else "Moderate" if a > 0.25 else "Borderline"


def explain(X_row, top=8):
    """Words pushing towards spam / ham for one message (linear model)."""
    if COEFS is None:
        return None
    row = X_row.toarray().ravel() if hasattr(X_row, "toarray") else np.asarray(X_row).ravel()
    contrib = row * COEFS
    idx = np.nonzero(row)[0]
    if len(idx) == 0:
        return None
    df = pd.DataFrame({"term": FEATURES[idx], "contribution": contrib[idx]})
    df["pushes towards"] = np.where(df["contribution"] < 0, "Spam", "Ham")
    df["strength"] = df["contribution"].abs()
    return df.sort_values("strength", ascending=False).head(top).reset_index(drop=True)


# ---------------------------------------------------------------- sidebar
EXAMPLES = {
    "🎁 Prize scam": "URGENT! You have won a 1-week free membership to our prize jackpot! Text CLAIM to 81010 to receive your reward!",
    "💳 Bank alert (fake)": "Dear customer, your account is suspended. Call 09061701461 now to claim your £2000 refund. T&Cs apply.",
    "📞 Premium offer": "FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! Subscribe6GBP/mnth",
    "💬 Normal chat": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "🏠 Casual ham": "Ok lar... Joking wif u oni... I'll reach home by 7 and call you.",
}

with st.sidebar:
    st.title("Spam Shield")
    page = st.radio("Navigate", ["🔍 Classify a message", "📂 Batch classification",
                                 "📊 Model & experiments", "📘 About the project"])
    st.divider()
    st.caption("Model: Linear SVM (C=10)\n\nFeatures: TF-IDF, 1-2 grams, 2,500 terms\n\nData: SMS Spam Collection")

# ------------------------------------------------------------ page: single
if page.startswith("🔍"):
    st.title("Spam SMS / Message Classifier")
    st.write("Paste a message and the trained SVM tells you whether it is **Spam** or **Ham** (legitimate).")

    if "msg" not in st.session_state:
        st.session_state.msg = ""

    st.write("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        if col.button(name):
            st.session_state.msg = text

    message = st.text_area("Message", key="msg", height=150,
                           placeholder="Type or paste an SMS / email text here...")

    if st.button("Classify", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            cleaned, X, scores, labels = predict([message])
            d, label = float(scores[0]), labels[0]

            if len(cleaned[0].split()) == 0:
                st.info("No usable words were found after cleaning (only numbers/symbols/stopwords), "
                        "so the result is unreliable.")

            c1, c2, c3 = st.columns(3)
            if label == "Spam":
                c1.error("🚨 **SPAM**")
            else:
                c1.success("✅ **HAM (not spam)**")
            c2.metric("Decision margin", f"{d:+.2f}",
                      help="Signed distance from the SVM boundary. Negative = spam, positive = ham.")
            c3.metric("Signal strength", strength(d))

            sp = float(spam_score(d))
            st.write("**Spam indicator** (margin-based, not a calibrated probability)")
            st.progress(min(max(sp, 0.0), 1.0), text=f"{sp * 100:.0f}% towards spam")

            exp = explain(X[0])
            if exp is not None and not exp.empty:
                st.subheader("Why this result?")
                st.caption("Terms found in your message that influenced the SVM most "
                           "(stemmed, as the model sees them).")
                chart_df = exp.set_index("term")[["contribution"]]
                left, right = st.columns([3, 2])
                left.bar_chart(chart_df, horizontal=True)
                right.dataframe(exp[["term", "pushes towards", "contribution"]].round(3),
                                hide_index=True)

            with st.expander("See the cleaned text the model received"):
                st.code(cleaned[0] or "(empty)")

# ------------------------------------------------------------- page: batch
elif page.startswith("📂"):
    st.title("📂 Batch classification")
    st.write("Upload a CSV with a text column, or paste one message per line.")

    tab_up, tab_paste = st.tabs(["Upload CSV", "Paste messages"])
    texts, source_df = None, None

    with tab_up:
        file = st.file_uploader("CSV file", type=["csv"])
        if file is not None:
            try:
                source_df = pd.read_csv(file, encoding="latin-1")
                col = st.selectbox("Column containing the message text", source_df.columns,
                                   index=min(1, len(source_df.columns) - 1))
                texts = source_df[col].fillna("").astype(str).tolist()
            except Exception as e:
                st.error(f"Could not read the file: {e}")

    with tab_paste:
        raw = st.text_area("One message per line", height=200)
        if raw.strip() and texts is None:
            texts = [l for l in raw.splitlines() if l.strip()]
            source_df = pd.DataFrame({"message": texts})

    if texts:
        if st.button("Run batch prediction", type="primary"):
            _, _, scores, labels = predict(texts)
            out = source_df.copy()
            out["prediction"] = labels
            out["margin"] = scores.round(3)
            out["strength"] = [strength(s) for s in scores]

            n_spam = int((labels == "Spam").sum())
            a, b, c = st.columns(3)
            a.metric("Messages", len(texts))
            b.metric("Spam", n_spam)
            c.metric("Ham", len(texts) - n_spam)

            st.dataframe(out)
            st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "spam_predictions.csv", "text/csv")

# ----------------------------------------------------------- page: model
elif page.startswith("📊"):
    st.title("📊 Model & experiments")
    st.write("Three text-representation techniques were tried, each with five classifiers "
             "(baseline, then tuned with GridSearchCV).")

    st.subheader("Baseline test accuracy (after tuning)")
    base = pd.DataFrame({
        "Model": ["Multinomial / Gaussian NB", "Logistic Regression", "Random Forest", "Extra Trees", "SVM"],
        "BoW": [0.9803, 0.9830, 0.9812, 0.9803, 0.9857],
        "TF-IDF": [0.9785, 0.9812, 0.9812, 0.9812, 0.9839],
        "Word2Vec (avg)": [0.5883, 0.9354, 0.9659, 0.9650, 0.9417],
    })
    st.dataframe(base, hide_index=True)
    st.bar_chart(base.set_index("Model")[["BoW", "TF-IDF", "Word2Vec (avg)"]])

    st.subheader("Deployed model")
    st.json({
        "type": "SVC (scikit-learn)", "kernel": str(model.kernel), "C": float(model.C),
        "support_vectors": int(model.n_support_.sum()),
        "vectorizer": "TfidfVectorizer(max_features=2500, ngram_range=(1,2))",
        "label_encoding": "True = Ham, False = Spam",
    })

    if COEFS is not None:
        st.subheader("Most influential words overall")
        order = np.argsort(COEFS)
        l, r = st.columns(2)
        l.write("**Strongest SPAM indicators**")
        l.dataframe(pd.DataFrame({"term": FEATURES[order[:15]], "weight": COEFS[order[:15]].round(2)}),
                    hide_index=True)
        r.write("**Strongest HAM indicators**")
        r.dataframe(pd.DataFrame({"term": FEATURES[order[::-1][:15]], "weight": COEFS[order[::-1][:15]].round(2)}),
                    hide_index=True)

# ------------------------------------------------------------ page: about
else:
    st.title("About the project")
    st.markdown("""
### Problem
Unwanted spam messages waste time and carry scams. This project builds a text classifier that labels a
message as **Spam** or **Ham** using the SMS Spam Collection dataset (5,572 labelled messages, about 13% spam).

### How the model was built
1. **Data loading & cleaning**: loaded `spam.csv` (latin-1), dropped three empty columns, kept `v1` (label) and `v2` (text).
2. **Text preprocessing**: removed non-letters, lowercased, removed English stopwords, applied Porter stemming.
3. **Target encoding**: `pd.get_dummies` on the label, so `True = Ham` and `False = Spam`.
4. **Train/test split**: 80/20. The vectorizer is fitted on training data only, to prevent data leakage.
5. **Three feature techniques**: Bag-of-Words, TF-IDF (both 2,500 features, 1-2 grams) and averaged Word2Vec (trained from scratch, 100 dims).
6. **Five classifiers per technique**: Naive Bayes, Logistic Regression, Random Forest, Extra Trees, SVM.
7. **Tuning**: GridSearchCV on each model, compared train vs test accuracy to watch for overfitting, plus confusion matrices.
8. **Selection**: the tuned linear SVM on TF-IDF was chosen for deployment (strong accuracy, fast inference, modest memory).
9. **Saving**: model and vectorizer saved with `joblib`, loaded here for predictions.

### Key takeaways
- TF-IDF/BoW with linear models clearly beat Word2Vec trained from scratch on this small dataset.
- Because the model is linear, each prediction can be explained by the words that drove it.
- Spam is the minority class, so check **spam recall / precision**, not just accuracy.

### Limitations
- Trained on English SMS from a specific period; modern scams and other languages may be missed.
- The "spam indicator" is derived from the SVM margin and is not a calibrated probability.
- Use as a decision aid, not a sole filter.
""")