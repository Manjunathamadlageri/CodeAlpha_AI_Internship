"""
Task 2: Chatbot for FAQs
CodeAlpha AI Internship

NLP Pipeline:
  1. Text preprocessing  — tokenization, lowercasing, stopword removal, stemming
  2. TF-IDF vectorization — sklearn TfidfVectorizer
  3. Cosine similarity   — sklearn cosine_similarity
  4. Intent matching     — top-K FAQ retrieval

Install: pip install scikit-learn nltk colorama
Run:     python task2_faq_chatbot.py
"""

import sys
import json
import math
import re
from collections import Counter

# ── Try rich imports; fall back gracefully ──────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    USE_SKLEARN = True
except ImportError:
    USE_SKLEARN = False

try:
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    try:
        STOPWORDS = set(stopwords.words("english"))
        STEMMER   = PorterStemmer()
        USE_NLTK  = True
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt",     quiet=True)
        STOPWORDS = set(stopwords.words("english"))
        STEMMER   = PorterStemmer()
        USE_NLTK  = True
except ImportError:
    USE_NLTK  = False
    STOPWORDS = {
        "i","me","my","we","our","you","your","he","his","she","her","it","its",
        "they","their","what","which","who","this","that","am","is","are","was",
        "were","be","been","have","has","had","do","does","did","a","an","the",
        "and","but","if","or","of","at","by","for","with","to","from","in","on",
        "how","can","will","would","could","should","not","no","about","also",
    }

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    USE_COLOR = True
except ImportError:
    USE_COLOR = False
    class _Mock:
        def __getattr__(self, _): return ""
    Fore = Style = _Mock()


# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════
FAQ_KB = [
    # ── General ─────────────────────────────────────────────────────────────
    {
        "question": "What is this chatbot?",
        "answer": "I'm AskBot, an NLP-powered FAQ assistant built for the CodeAlpha AI Internship (Task 2). I use TF-IDF vectorization and cosine similarity to match your question to the most relevant FAQ.",
        "category": "general"
    },
    {
        "question": "How does the chatbot work?",
        "answer": "My NLP pipeline: (1) Preprocess your input via tokenization, stopword removal & stemming. (2) Vectorize all FAQ questions using TF-IDF. (3) Compute cosine similarity between your query vector and all FAQ vectors. (4) Return the highest-scoring match.",
        "category": "general"
    },
    {
        "question": "What topics can I ask about?",
        "answer": "I cover four categories: General (about this bot), Account (sign-up, login, password), Technical (browser support, security), and Billing (payments, refunds, plans).",
        "category": "general"
    },

    # ── Account ──────────────────────────────────────────────────────────────
    {
        "question": "How do I create an account?",
        "answer": "Click 'Sign Up', enter your name, email, and a strong password. Check your inbox for a verification email and click the activation link. Done in under 2 minutes!",
        "category": "account"
    },
    {
        "question": "How do I reset or change my password?",
        "answer": "On the login page, click 'Forgot Password', enter your registered email, and follow the link sent to your inbox (valid for 30 minutes). Check spam if you don't see it.",
        "category": "account"
    },
    {
        "question": "How do I update my profile?",
        "answer": "Go to Settings → Profile. You can update your name, email, and avatar. Email changes require re-verification.",
        "category": "account"
    },
    {
        "question": "How do I delete my account?",
        "answer": "Settings → Account → Delete Account. Confirm with your password. All data is removed within 30 days. Export your data first if needed.",
        "category": "account"
    },

    # ── Technical ────────────────────────────────────────────────────────────
    {
        "question": "Which browsers are supported?",
        "answer": "We support the latest two versions of Chrome, Firefox, Safari, and Edge. Internet Explorer is not supported. Enable JavaScript and cookies for full functionality.",
        "category": "technical"
    },
    {
        "question": "Is there a mobile app available?",
        "answer": "Yes! Download our app from the iOS App Store or Google Play. Requires iOS 14+ or Android 10+. Supports offline mode and push notifications.",
        "category": "technical"
    },
    {
        "question": "How is my data kept secure and private?",
        "answer": "We use AES-256 encryption at rest and TLS 1.3 in transit. Compliant with GDPR, ISO 27001, and SOC 2 Type II. Your data is never sold to third parties.",
        "category": "technical"
    },
    {
        "question": "The app is slow, what should I do?",
        "answer": "Try: (1) Check your internet speed, (2) Clear browser cache, (3) Close unused tabs, (4) Disable ad blockers for this site. Still slow? Contact support.",
        "category": "technical"
    },

    # ── Billing ──────────────────────────────────────────────────────────────
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, Mastercard, American Express, RuPay, UPI, Net Banking, and PayPal — all processed through PCI-DSS compliant gateways.",
        "category": "billing"
    },
    {
        "question": "Can I get a refund?",
        "answer": "Annual plans: refunds available within 7 days if premium features are unused. Monthly plans are non-refundable but cancellable. Email billing@example.com with your order ID.",
        "category": "billing"
    },
    {
        "question": "How do I cancel my subscription?",
        "answer": "Settings → Billing → Cancel Subscription. Access continues until the end of your billing period. No cancellation fees.",
        "category": "billing"
    },
    {
        "question": "What is included in the free plan?",
        "answer": "Free plan includes: 5 projects, 1GB storage, community support, and core features. Paid plans unlock unlimited projects, API access, and priority support.",
        "category": "billing"
    },
    {
        "question": "Are there student discounts available?",
        "answer": "Yes! Students receive 50% off all plans. Verify via .edu email or student ID at Settings → Billing → Student Discount. Renews annually.",
        "category": "billing"
    },
]


# ═══════════════════════════════════════════════════════════════════
#  NLP ENGINE
# ═══════════════════════════════════════════════════════════════════

def preprocess(text: str) -> str:
    """Lowercase → remove punctuation → remove stopwords → (optionally) stem."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if USE_NLTK:
        tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


class FAQChatbot:
    def __init__(self, faq_data: list):
        self.faq = faq_data
        self.processed_questions = [preprocess(f["question"] + " " + f["answer"]) for f in faq_data]
        self._build_index()

    def _build_index(self):
        if USE_SKLEARN:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)
        else:
            # Pure-Python fallback TF-IDF
            self._build_manual_tfidf()

    # ── Pure-Python TF-IDF ──────────────────────────────────────────────────
    def _build_manual_tfidf(self):
        docs = self.processed_questions
        N = len(docs)
        df = Counter()
        tokenized = [d.split() for d in docs]
        for tokens in tokenized:
            for t in set(tokens):
                df[t] += 1
        self._idf = {t: math.log((N+1)/(c+1))+1 for t,c in df.items()}
        self._tfidf_vecs = [self._tfidf_vec(tokens) for tokens in tokenized]

    def _tfidf_vec(self, tokens):
        tf = Counter(tokens)
        n = len(tokens) or 1
        return {t: (c/n) * self._idf.get(t, 1.0) for t, c in tf.items()}

    def _cosine(self, a, b):
        dot = sum(a.get(k,0)*b.get(k,0) for k in a)
        na  = math.sqrt(sum(v*v for v in a.values()))
        nb  = math.sqrt(sum(v*v for v in b.values()))
        return dot/(na*nb) if na and nb else 0.0

    # ── Core matching ───────────────────────────────────────────────────────
    def match(self, query: str, top_k: int = 3):
        pq = preprocess(query)
        if not pq.strip():
            return []

        if USE_SKLEARN:
            q_vec = self.vectorizer.transform([pq])
            sims  = cosine_similarity(q_vec, self.tfidf_matrix)[0]
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        else:
            q_tokens = pq.split()
            q_vec    = self._tfidf_vec(q_tokens)
            sims     = [self._cosine(q_vec, v) for v in self._tfidf_vecs]
            ranked   = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            if score > 0.02:
                results.append({**self.faq[idx], "score": float(score)})
        return results

    def respond(self, query: str):
        results = self.match(query, top_k=3)
        if not results:
            return {
                "answer": "I'm not sure about that. Try rephrasing, or ask about: account setup, billing, technical support, or security.",
                "confidence": 0.0,
                "category": "unknown",
                "suggestions": [f["question"] for f in self.faq[:3]],
            }
        best = results[0]
        suggs = [r["question"] for r in results[1:3]]
        return {
            "answer": best["answer"],
            "confidence": best["score"],
            "category": best["category"],
            "question_matched": best["question"],
            "suggestions": suggs,
        }


# ═══════════════════════════════════════════════════════════════════
#  CLI CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════

def banner():
    print(f"""
{Fore.CYAN}{'═'*55}{Style.RESET_ALL}
{Fore.WHITE}   🤖  AskBot — FAQ Chatbot with NLP{Style.RESET_ALL}
{Fore.WHITE}       CodeAlpha AI Internship — Task 2{Style.RESET_ALL}
{Fore.CYAN}{'─'*55}{Style.RESET_ALL}
{Fore.YELLOW}  Engine : {'scikit-learn TF-IDF + cosine_similarity' if USE_SKLEARN else 'Pure-Python TF-IDF (no sklearn)'}
  NLP    : {'NLTK tokenization + PorterStemmer' if USE_NLTK else 'Basic tokenization (no nltk)'}{Style.RESET_ALL}
{Fore.CYAN}{'═'*55}{Style.RESET_ALL}
Type your question (or 'help', 'list', 'quit').
""")


def conf_bar(score: float, width=20):
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = score * 100
    if pct >= 60:
        color = Fore.GREEN
    elif pct >= 30:
        color = Fore.YELLOW
    else:
        color = Fore.RED
    return f"{color}[{bar}] {pct:5.1f}%{Style.RESET_ALL}"


def main():
    banner()
    bot = FAQChatbot(FAQ_KB)
    print(f"{Fore.CYAN}Bot:{Style.RESET_ALL} Hello! Ask me anything about accounts, billing, technical support, or AI concepts.\n")

    while True:
        try:
            user_input = input(f"{Fore.GREEN}You:{Style.RESET_ALL} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print(f"{Fore.CYAN}Bot:{Style.RESET_ALL} Goodbye! 👋")
            break
        if user_input.lower() == "list":
            print(f"\n{Fore.YELLOW}Available FAQs:{Style.RESET_ALL}")
            for i, f in enumerate(FAQ_KB, 1):
                print(f"  {i:2}. [{f['category']:9}] {f['question']}")
            print()
            continue
        if user_input.lower() in ("help", "?"):
            print(f"""
{Fore.YELLOW}Commands:{Style.RESET_ALL}
  list   — show all FAQ questions
  quit   — exit the chatbot
  help   — show this message
  (anything else) — ask a question!
""")
            continue

        response = bot.respond(user_input)

        print(f"\n{Fore.CYAN}Bot:{Style.RESET_ALL} {response['answer']}")
        print(f"    {Fore.MAGENTA}Category:{Style.RESET_ALL} {response['category'].title()}", end="")
        if "question_matched" in response:
            print(f"  |  {Fore.MAGENTA}Matched:{Style.RESET_ALL} \"{response['question_matched']}\"", end="")
        print()
        print(f"    {Fore.MAGENTA}Confidence:{Style.RESET_ALL} {conf_bar(response['confidence'])}")

        if response.get("suggestions"):
            print(f"\n    {Fore.YELLOW}You might also ask:{Style.RESET_ALL}")
            for s in response["suggestions"]:
                print(f"      • {s}")
        print()


if __name__ == "__main__":
    main()
