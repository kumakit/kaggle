import re
import numpy as np
import pandas as pd

def extract_stylometry_features(df, text_col="text"):
    feats = pd.DataFrame(index=df.index)
    texts = df[text_col].astype(str)
    
    # 1. Length & Basic Counts
    char_lens = texts.str.len()
    words_list = texts.apply(lambda t: t.split())
    word_counts = words_list.apply(len)
    
    feats["char_len"] = char_lens
    feats["word_count"] = word_counts
    feats["mean_word_len"] = char_lens / (word_counts + 1e-5)
    
    # Type-Token Ratio (Vocabulary Diversity)
    feats["ttr"] = words_list.apply(lambda ws: len(set(ws)) / (len(ws) + 1e-5))
    
    # Uppercase & Digits
    feats["upper_count"] = texts.apply(lambda t: sum(1 for c in t if c.isupper()))
    feats["upper_ratio"] = feats["upper_count"] / (char_lens + 1e-5)
    feats["digit_count"] = texts.apply(lambda t: sum(1 for c in t if c.isdigit()))
    feats["digit_ratio"] = feats["digit_count"] / (char_lens + 1e-5)
    
    # 2. Typography: Quotes
    feats["cnt_straight_single"] = texts.apply(lambda t: t.count("'"))
    feats["cnt_curly_single_open"] = texts.apply(lambda t: t.count("‘"))
    feats["cnt_curly_single_close"] = texts.apply(lambda t: t.count("’"))
    feats["cnt_curly_single"] = feats["cnt_curly_single_open"] + feats["cnt_curly_single_close"]
    feats["ratio_curly_single"] = feats["cnt_curly_single"] / (feats["cnt_straight_single"] + feats["cnt_curly_single"] + 1e-5)
    
    feats["cnt_straight_double"] = texts.apply(lambda t: t.count('"'))
    feats["cnt_curly_double_open"] = texts.apply(lambda t: t.count("“"))
    feats["cnt_curly_double_close"] = texts.apply(lambda t: t.count("”"))
    feats["cnt_curly_double"] = feats["cnt_curly_double_open"] + feats["cnt_curly_double_close"]
    feats["ratio_curly_double"] = feats["cnt_curly_double"] / (feats["cnt_straight_double"] + feats["cnt_curly_double"] + 1e-5)
    
    # 3. Typography: Dashes & Hyphens
    feats["cnt_hyphen"] = texts.apply(lambda t: t.count("-"))
    feats["cnt_en_dash"] = texts.apply(lambda t: t.count("–"))
    feats["cnt_em_dash"] = texts.apply(lambda t: t.count("—"))
    # Unspaced em dash (e.g. "word—word") vs spaced ("word — word")
    feats["cnt_unspaced_em_dash"] = texts.apply(lambda t: len(re.findall(r"[^\s]—[^\s]", t)))
    feats["cnt_spaced_em_dash"] = texts.apply(lambda t: len(re.findall(r"\s—\s", t)))
    
    # 4. Punctuations
    feats["cnt_comma"] = texts.apply(lambda t: t.count(","))
    feats["cnt_period"] = texts.apply(lambda t: t.count("."))
    feats["cnt_colon"] = texts.apply(lambda t: t.count(":"))
    feats["cnt_semicolon"] = texts.apply(lambda t: t.count(";"))
    feats["cnt_exclamation"] = texts.apply(lambda t: t.count("!"))
    feats["cnt_question"] = texts.apply(lambda t: t.count("?"))
    feats["cnt_ellipsis"] = texts.apply(lambda t: len(re.findall(r"\.\.\.", t)))
    
    # Ratios per 1000 chars
    for p in ["comma", "period", "colon", "semicolon", "exclamation", "question"]:
        feats[f"ratio_{p}"] = feats[f"cnt_{p}"] / (char_lens + 1e-5) * 1000
    
    # 5. Structure & Layout
    feats["cnt_newlines"] = texts.apply(lambda t: t.count("\n"))
    feats["cnt_paragraphs"] = texts.apply(lambda t: len([p for p in t.split("\n\n") if p.strip()]))
    feats["has_bullet"] = texts.apply(lambda t: int(bool(re.search(r"(?:^|\n)\s*[-*•]\s", t))))
    feats["has_numbered_list"] = texts.apply(lambda t: int(bool(re.search(r"(?:^|\n)\s*\d+\.\s", t))))
    feats["cnt_backticks"] = texts.apply(lambda t: t.count("`"))
    feats["cnt_asterisks"] = texts.apply(lambda t: t.count("*"))
    feats["has_bold_markdown"] = texts.apply(lambda t: int(bool(re.search(r"\*\*[^*]+\*\*", t))))
    
    # 6. Sentence-level statistics
    def sentence_stats(text):
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sents:
            return 0, 0.0, 0.0
        lens = [len(s.split()) for s in sents]
        return len(sents), np.mean(lens), np.std(lens)
        
    s_stats = texts.apply(sentence_stats)
    feats["sent_count"] = s_stats.apply(lambda x: x[0])
    feats["mean_sent_word_len"] = s_stats.apply(lambda x: x[1])
    feats["std_sent_word_len"] = s_stats.apply(lambda x: x[2])
    
    # 7. Common Discourse Markers (Frequency per text)
    markers = [
        "however", "moreover", "furthermore", "thus", "therefore",
        "additionally", "overall", "in summary", "crucially", "notably",
        "indeed", "first", "second", "finally", "specifically"
    ]
    for m in markers:
        col = f"dm_{m.replace(' ', '_')}"
        feats[col] = texts.apply(lambda t, pat=m: len(re.findall(r"\b" + pat + r"\b", t, re.IGNORECASE)))
        
    return feats

if __name__ == "__main__":
    train = pd.read_csv("005_name-that-llm/data/train.csv")
    feats = extract_stylometry_features(train)
    feats["model"] = train["model"]
    
    print(f"Extracted {feats.shape[1] - 1} stylometry features.")
    
    # Check features with highest variance across models
    means_by_model = feats.groupby("model").mean()
    stds_across_models = means_by_model.std().sort_values(ascending=False)
    
    print("\n=== Top 20 Most Discriminative Stylometry Features (Std across model means) ===")
    print(stds_across_models.head(20))
    
    # Let's inspect some specific key features
    print("\n=== Unspaced Em Dash (cnt_unspaced_em_dash) by Model ===")
    print(means_by_model["cnt_unspaced_em_dash"].round(3))
    
    print("\n=== Straight Double Quote (cnt_straight_double) by Model ===")
    print(means_by_model["cnt_straight_double"].round(3))
    
    print("\n=== Semicolon (cnt_semicolon) by Model ===")
    print(means_by_model["cnt_semicolon"].round(3))
    
    print("\n=== Mean Word Length (mean_word_len) by Model ===")
    print(means_by_model["mean_word_len"].round(3))
    
    print("\n=== Character Length (char_len) by Model ===")
    print(means_by_model["char_len"].round(1))
