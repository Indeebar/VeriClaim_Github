from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

_model             = None
_pattern_embeddings = None
_patterns          = None
_keywords          = None


def load_nlp_model(
    patterns_path='models/claim_nlp/fraud_patterns.json'
):
    global _model, _pattern_embeddings, _patterns, _keywords

    _model = SentenceTransformer('all-MiniLM-L6-v2')

    with open(patterns_path, 'r') as f:
        data = json.load(f)

    _patterns = data['high_risk_patterns']
    _keywords = data['high_risk_keywords']

    _pattern_embeddings = _model.encode(
        _patterns,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    print(f'[NLP] Loaded {len(_patterns)} fraud patterns and '
          f'{len(_keywords)} keywords')


def embed_text(text: str) -> np.ndarray:
    if _model is None:
        raise RuntimeError(
            'NLP model not loaded. Call load_nlp_model() first.'
        )
    return _model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]


def get_pattern_embeddings():
    return _pattern_embeddings


def get_patterns():
    return _patterns


def get_keywords():
    return _keywords