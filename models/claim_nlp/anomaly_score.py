import numpy as np
from models.claim_nlp.embed import (
    embed_text,
    get_pattern_embeddings,
    get_patterns,
    get_keywords
)

KEYWORD_WEIGHTS = {
    'total loss':      0.40,
    'fire':            0.30,
    'stolen':          0.30,
    'no witnesses':    0.35,
    'no cctv':         0.40,
    'fled scene':      0.25,
    'remote location': 0.20,
    'deserted':        0.20,
    'overnight':       0.15,
    'basement parking':0.20,
    '3am':             0.25,
    'no cameras':      0.35,
    'spontaneous':     0.25,
    'unknown vehicle': 0.20,
    'documents lost':  0.30,
    'no police report':0.30,
    'unseasonal':      0.20,
    'submerged':       0.25,
    'brake failure':   0.20,
}


def score_text(incident_text: str) -> dict:
    """
    Score a free-text incident description for fraud signals.

    Returns dict with:
        anomaly_score      : float 0.0 to 1.0 (higher = more suspicious)
        semantic_score     : float (cosine similarity to fraud patterns)
        keyword_score      : float (weighted keyword matches)
        triggered_keywords : list of matched keywords
        top_fraud_pattern  : the closest matching known fraud pattern
    """
    if not incident_text or len(incident_text.strip()) < 3:
        return {
            'anomaly_score':      0.0,
            'semantic_score':     0.0,
            'keyword_score':      0.0,
            'triggered_keywords': [],
            'top_fraud_pattern':  None
        }

    text_lower = incident_text.lower().strip()

    # Layer 1 — semantic similarity
    pattern_embeddings = get_pattern_embeddings()
    patterns           = get_patterns()

    if pattern_embeddings is not None:
        text_emb  = embed_text(incident_text)
        sims      = np.dot(pattern_embeddings, text_emb)
        max_sim   = float(sims.max())
        top_idx   = int(sims.argmax())
        top_pattern = patterns[top_idx] if max_sim > 0.3 else None
    else:
        max_sim     = 0.0
        top_pattern = None

    # Layer 2 — keyword scan
    keyword_score    = 0.0
    triggered        = []
    for kw, weight in KEYWORD_WEIGHTS.items():
        if kw in text_lower:
            keyword_score = min(1.0, keyword_score + weight)
            triggered.append(kw)

    # Also check keywords from json file
    extra_keywords = get_keywords() or []
    for kw in extra_keywords:
        if kw in text_lower and kw not in triggered:
            keyword_score = min(1.0, keyword_score + 0.15)
            triggered.append(kw)

    # Combine: semantic 60% + keyword 40%
    combined = min(1.0, (max_sim * 0.6) + (keyword_score * 0.4))

    return {
        'anomaly_score':      round(combined,      4),
        'semantic_score':     round(max_sim,        4),
        'keyword_score':      round(keyword_score,  4),
        'triggered_keywords': triggered,
        'top_fraud_pattern':  top_pattern
    }