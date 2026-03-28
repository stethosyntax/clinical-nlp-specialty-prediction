"""
src/preprocess.py
-----------------
Reusable text preprocessing utilities for clinical NLP pipeline.
Author: Aruna Kunche
"""

import re
import spacy
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
for resource in ['stopwords', 'wordnet', 'punkt']:
    nltk.download(resource, quiet=True)

# Load spaCy model (disable unused components for speed)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Standard English stopwords + clinical noise words
STOP_WORDS = set(stopwords.words('english'))
CLINICAL_STOPWORDS = {
    'patient', 'history', 'procedure', 'noted', 'performed',
    'placed', 'using', 'also', 'well', 'used', 'left', 'right',
    'normal', 'without', 'within', 'year', 'old', 'mg', 'mm',
    'cm', 'time', 'day', 'one', 'two', 'three', 'given', 'seen',
    'follow', 'taken', 'make', 'showed'
}
STOP_WORDS.update(CLINICAL_STOPWORDS)


def preprocess_clinical_text(text: str) -> str:
    """
    Clean and normalize a clinical note for NLP modeling.

    Pipeline:
        1. Lowercase
        2. Remove numbers and measurements
        3. Remove punctuation and special characters
        4. Collapse whitespace
        5. Tokenize with spaCy
        6. Remove stopwords (English + clinical)
        7. Lemmatize

    Parameters
    ----------
    text : str
        Raw clinical transcription note.

    Returns
    -------
    str
        Cleaned, lemmatized string of meaningful medical terms.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ''

    # 1. Lowercase
    text = text.lower()

    # 2. Remove numbers and measurements (e.g., 5mg, 120/80)
    text = re.sub(r'\d+\.?\d*', '', text)

    # 3. Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 4. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 5-7. Tokenize, filter stopwords, lemmatize
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.text not in STOP_WORDS
        and len(token.text) > 2
        and not token.is_space
        and token.is_alpha
    ]

    return ' '.join(tokens)


def batch_preprocess(texts, verbose: bool = True) -> list:
    """
    Apply preprocess_clinical_text to a list or Series of notes.

    Parameters
    ----------
    texts : list or pd.Series
        Collection of raw clinical notes.
    verbose : bool
        Print progress message.

    Returns
    -------
    list of str
        Cleaned notes.
    """
    if verbose:
        print(f'Preprocessing {len(texts):,} clinical notes...')
    cleaned = [preprocess_clinical_text(t) for t in texts]
    if verbose:
        print('Done.')
    return cleaned
