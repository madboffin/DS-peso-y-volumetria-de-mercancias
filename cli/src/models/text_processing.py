"""Text processing utilities for feature extraction."""

from typing import List

import numpy as np
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")


def get_text_features(texts: List[str], batch_size: int = 1000) -> np.ndarray:
    """Convert texts to spaCy vectors.

    Args:
        texts: List of text strings to process
        batch_size: Number of texts to process at once

    Returns:
        Array of text vectors
    """
    vectors = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        if doc.has_vector:
            vectors.append(doc.vector)
        else:
            vectors.append(np.zeros(nlp.vocab.vectors.shape[1]))
    return np.array(vectors)
