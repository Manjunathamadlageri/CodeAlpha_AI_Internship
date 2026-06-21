"""
Task 1: Language Translation Tool
CodeAlpha AI Internship

Uses MyMemory (free, no API key required) or optionally Google Translate API.
Install: pip install requests deep-translator
Run:     python task1_translation.py
"""

import requests

# ─── Optional: use deep-translator for richer API support ───────────────────
try:
    from deep_translator import GoogleTranslator, single_detection
    USE_DEEP_TRANSLATOR = True
except ImportError:
    USE_DEEP_TRANSLATOR = False

SUPPORTED_LANGUAGES = {
    "en": "English",   "hi": "Hindi",     "kn": "Kannada",
    "te": "Telugu",    "ta": "Tamil",     "ml": "Malayalam",
    "fr": "French",    "de": "German",    "es": "Spanish",
    "it": "Italian",   "pt": "Portuguese","ru": "Russian",
    "zh": "Chinese",   "ja": "Japanese",  "ko": "Korean",
    "ar": "Arabic",    "tr": "Turkish",   "nl": "Dutch",
    "pl": "Polish",    "sv": "Swedish",   "bn": "Bengali",
}


def translate_mymemory(text: str, source: str, target: str) -> dict:
    """Translate using the free MyMemory API (no key required)."""
    if not text.strip():
        return {"success": False, "error": "Empty input"}

    lang_pair = f"{source}|{target}" if source != "auto" else f"|{target}"
    url = "https://api.mymemory.translated.net/get"
    params = {"q": text, "langpair": lang_pair}

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data.get("responseStatus") in (200, "200"):
            return {
                "success": True,
                "translated": data["responseData"]["translatedText"],
                "confidence": data["responseData"].get("match", 0),
                "detected": data["responseData"].get("detectedLanguage", source),
            }
        return {"success": False, "error": data.get("responseDetails", "Unknown error")}

    except requests.RequestException as e:
        return {"success": False, "error": str(e)}


def translate_deep(text: str, source: str, target: str) -> dict:
    """Translate using deep-translator (wraps Google Translate)."""
    try:
        src = "auto" if source == "auto" else source
        result = GoogleTranslator(source=src, target=target).translate(text)
        return {"success": True, "translated": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def translate(text: str, source: str = "auto", target: str = "hi") -> dict:
    """Main translation function — uses deep-translator if available, else MyMemory."""
    if USE_DEEP_TRANSLATOR:
        return translate_deep(text, source, target)
    return translate_mymemory(text, source, target)


def print_languages():
    print("\n📋 Supported Language Codes:")
    for code, name in SUPPORTED_LANGUAGES.items():
        print(f"  {code:>4}  →  {name}")
    print()


def main():
    print("=" * 55)
    print("   🌐  LinguaAI — Language Translation Tool")
    print("         CodeAlpha AI Internship — Task 1")
    print("=" * 55)

    translation_history = []

    while True:
        print("\n[1] Translate  [2] View History  [L] Languages  [Q] Quit")
        choice = input("Choice: ").strip().lower()

        if choice == 'q':
            print("Goodbye! 👋")
            break

        elif choice == 'l':
            print_languages()

        elif choice == '2':
            if not translation_history:
                print("  No history yet.")
            else:
                print("\n📜 Translation History:")
                for i, entry in enumerate(translation_history, 1):
                    print(f"  {i}. [{entry['src_lang']} → {entry['tgt_lang']}]")
                    print(f"     Source : {entry['source'][:60]}")
                    print(f"     Result : {entry['result'][:60]}")

        elif choice == '1':
            print(f"\n  Engine: {'deep-translator (Google)' if USE_DEEP_TRANSLATOR else 'MyMemory API'}")
            text = input("  Enter text to translate: ").strip()
            if not text:
                print("  ⚠ No text entered.")
                continue

            print(f"\n  Source language (default: auto) — type 'L' to list codes")
            src = input("  Source lang code: ").strip().lower() or "auto"
            if src == 'l':
                print_languages()
                src = input("  Source lang code: ").strip().lower() or "auto"

            tgt = input("  Target lang code (default: hi): ").strip().lower() or "hi"
            if tgt == 'l':
                print_languages()
                tgt = input("  Target lang code: ").strip().lower() or "hi"

            print("\n  ⏳ Translating...")
            result = translate(text, src, tgt)

            if result["success"]:
                print(f"\n  ✅ Translation ({SUPPORTED_LANGUAGES.get(src,src)} → {SUPPORTED_LANGUAGES.get(tgt,tgt)}):")
                print(f"  {result['translated']}")
                if "confidence" in result:
                    print(f"  Confidence: {result['confidence']:.0%}")
                if "detected" in result and src == "auto":
                    print(f"  Detected source: {result['detected']}")

                translation_history.append({
                    "source": text,
                    "result": result["translated"],
                    "src_lang": src,
                    "tgt_lang": tgt,
                })
            else:
                print(f"\n  ❌ Error: {result['error']}")
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
