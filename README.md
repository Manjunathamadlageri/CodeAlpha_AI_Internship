<<<<<<< HEAD
# CodeAlpha_AI_Internship 🤖

> **CodeAlpha Artificial Intelligence Internship** — Tasks 1 & 2  
> Submitted by **Manjunath Amad Lageri**  
> M.Tech AI & Data Science — MVJ College of Engineering, Bengaluru

---

## ✅ Task 1 — Language Translation Tool

### Overview
A full-featured language translation application supporting 25+ languages with automatic language detection, text-to-speech, copy functionality, and translation history.

### Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend (CLI) | Python · `requests` · `deep-translator` |
| Frontend (Web) | HTML5 · CSS3 · Vanilla JS |
| Translation API | MyMemory (free, no key required) / Google Translate |

### Features
- 🌐 25+ languages including all major Indian languages (Hindi, Kannada, Telugu, Tamil, etc.)
- 🔍 Auto language detection
- 🔊 Text-to-Speech for both source and translated text (Web Speech API)
- ⧉ One-click copy to clipboard
- ⇄ Swap source/target languages
- 📜 Local storage-backed translation history (last 10)
- ⌨️ Ctrl+Enter keyboard shortcut

### Files
```
task1_translation.py          ← Python CLI tool
task1_language_translator.html ← Standalone web UI (open in browser)
```

### How to Run (Python)
```bash
pip install requests deep-translator
python task1_translation.py
```

### How to Run (Web)
Simply open `task1_language_translator.html` in any modern browser — no server required.

---

## ✅ Task 2 — FAQ Chatbot with NLP

### Overview
An NLP-powered FAQ chatbot that uses **TF-IDF vectorization** and **cosine similarity** to match user questions against a curated knowledge base — no external API required.

### NLP Pipeline
```
User Input
    │
    ▼
[1] Preprocessing
    • Lowercase conversion
    • Punctuation removal
    • Stopword filtering (NLTK)
    • Porter Stemming (NLTK)
    │
    ▼
[2] TF-IDF Vectorization
    • scikit-learn TfidfVectorizer (ngram_range=(1,2))
    • Fitted on FAQ questions + answers corpus
    │
    ▼
[3] Cosine Similarity
    • sklearn.metrics.pairwise.cosine_similarity
    • Query vector vs. all FAQ vectors
    │
    ▼
[4] Top-K Retrieval
    • Best match → answer
    • Next 2 matches → suggestions
    • Confidence score displayed
```

### Tech Stack
| Layer | Technology |
|-------|-----------|
| NLP | `scikit-learn` · `nltk` (with pure-Python fallback) |
| Similarity | Cosine Similarity (TF-IDF space) |
| Frontend (Web) | HTML5 · CSS3 · Vanilla JS |

### FAQ Knowledge Base
The chatbot covers **4 categories** with **20+ FAQ entries**:

| Category | Topics |
|----------|--------|
| 🏢 General | About the bot, how it works |
| 👤 Account | Sign-up, password reset, profile, deletion |
| ⚙️ Technical | Browsers, mobile app, security, performance |
| 💳 Billing | Payments, refunds, cancellation, plans, discounts |

### Files
```
task2_faq_chatbot.py      ← Python CLI chatbot (full NLP pipeline)
task2_faq_chatbot.html    ← Standalone web UI with chat interface
```

### How to Run (Python)
```bash
pip install scikit-learn nltk colorama
python task2_faq_chatbot.py
```
> **Note:** If `scikit-learn`/`nltk` are not installed, the script automatically falls back to a pure-Python TF-IDF implementation — no dependencies needed.

### How to Run (Web)
Open `task2_faq_chatbot.html` in any modern browser — no server required. The NLP engine runs entirely in JavaScript (TF-IDF + cosine similarity reimplemented in JS).

---

## 📁 Repository Structure

```
CodeAlpha_AI_Internship/
├── Task1_LanguageTranslation/
│   ├── task1_translation.py
│   └── task1_language_translator.html
├── Task2_FAQChatbot/
│   ├── task2_faq_chatbot.py
│   └── task2_faq_chatbot.html
└── README.md
```

---

## 👤 Author

**Manjunath Amad Lageri**  
🎓 M.Tech AI & Data Science — MVJ College of Engineering  
🔗 GitHub: [github.com/Manjunathamadlageri](https://github.com/Manjunathamadlageri)  
📍 Bengaluru, India

---

*Built with ❤️ for the CodeAlpha AI Internship Program*
=======
# CodeAlpha_AI_Internship
>>>>>>> 30101e9d673ed686058005c3b45b9b6e11819b82
