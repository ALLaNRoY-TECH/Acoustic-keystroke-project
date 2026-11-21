"""
Stage E — Language-model decoding.

Lifts per-key accuracy to word-level recovery by treating the classifier's
probability outputs as a noisy channel and correcting against a dictionary.

This mirrors the classic Zhuang et al. approach: the LM does not need to know
the acoustic model — it just sees a sequence of per-character distributions and
finds the most probable word from a dictionary under the noisy-channel model.

Can be used stand-alone or imported by demo.py.
"""

import numpy as np
from config import KEY_TO_IDX, IDX_TO_KEY, KEYS


# ── Greedy baseline ───────────────────────────────────────────────────────────

def greedy_decode(probs_seq):
    """
    Argmax at each position.  probs_seq: list of (NUM_CLASSES,) arrays.
    Returns a string.
    """
    return ''.join(IDX_TO_KEY[int(np.argmax(p))] for p in probs_seq)


# ── Word-level correction ─────────────────────────────────────────────────────

def best_word(probs_seq, word_list):
    """
    Given per-character probability arrays for a single word-length segment,
    find the highest log-probability match in word_list.

    Time: O(|word_list|) per call — fine for offline use with 50k-word lists.
    """
    log_probs = [np.log(p + 1e-12) for p in probs_seq]
    n         = len(probs_seq)
    best, best_score = None, -np.inf

    for word in word_list:
        if len(word) != n:
            continue
        if not all(c in KEY_TO_IDX for c in word):
            continue
        score = sum(log_probs[i][KEY_TO_IDX[c]] for i, c in enumerate(word))
        if score > best_score:
            best, best_score = word, score

    return best


def decode_words(probs_seq, word_list=None):
    """
    Decode a sequence of per-key distributions into corrected text.

    Strategy:
      1. Greedy decode to get a rough character sequence.
      2. Split on greedy spaces (if 'space' key were in vocabulary — it isn't
         in the 36-key setup, so we use silence gaps instead; for now we
         split by known word boundaries from the greedy string).
      3. For each greedy word, search the dictionary for the best match.

    In the 36-key (a-z, 0-9) vocabulary there is no space key, so word
    boundaries must come from an oracle (e.g., you know the sentence structure)
    or from a CTC-style decoder.  Here we return per-word corrections given
    explicit word-length hints, which is useful for the password-recovery demo
    where you know the word lengths.
    """
    if word_list is None:
        word_list = load_word_list()

    greedy = greedy_decode(probs_seq)
    return greedy, word_list   # caller decides how to segment into words


def correct_word(probs_seq, word_list=None):
    """
    Single-word correction: the caller already knows this segment is one word.
    Returns the corrected word or the greedy string if no match found.
    """
    if word_list is None:
        word_list = load_word_list()
    result = best_word(probs_seq, word_list)
    return result if result else greedy_decode(probs_seq)


# ── Word list loading ─────────────────────────────────────────────────────────

def load_word_list(path=None):
    """
    Load a word list (lowercase, alpha-only).
    Priority: custom file -> nltk.corpus.words -> built-in fallback.
    """
    if path:
        with open(path) as f:
            return [w.strip().lower() for w in f if w.strip().isalpha()]

    try:
        import nltk
        try:
            words = nltk.corpus.words.words()
        except Exception:
            nltk.download('words', quiet=True)
            words = nltk.corpus.words.words()
        nltk_words = [w.lower() for w in words if w.isalpha() and len(w) <= 15]
        # Merge with the built-in list so common words (e.g. 'hello') are never
        # missing, regardless of which dictionary edition nltk ships.
        return sorted(set(nltk_words) | set(_FALLBACK_WORDS))
    except Exception:
        # No nltk, or the corpus could not be downloaded (e.g. offline).
        return list(_FALLBACK_WORDS)


# Built-in fallback — most common English words.
_FALLBACK_WORDS = [
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
        'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
        'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'yes', 'yet',
        'also', 'back', 'call', 'come', 'down', 'each', 'even', 'find', 'from',
        'give', 'good', 'have', 'help', 'here', 'high', 'home', 'just', 'keep',
        'kind', 'know', 'last', 'life', 'like', 'line', 'live', 'long', 'look',
        'made', 'make', 'many', 'mean', 'more', 'most', 'move', 'much', 'must',
        'name', 'need', 'next', 'open', 'over', 'part', 'play', 'real', 'same',
        'seem', 'show', 'side', 'some', 'such', 'take', 'talk', 'tell', 'than',
        'that', 'them', 'then', 'they', 'this', 'time', 'turn', 'very', 'want',
        'well', 'went', 'were', 'what', 'when', 'with', 'word', 'work', 'year',
        'your', 'about', 'after', 'again', 'along', 'begin', 'being', 'below',
        'bring', 'build', 'carry', 'check', 'chose', 'clean', 'clear', 'close',
        'could', 'cover', 'cross', 'drive', 'early', 'eight', 'enter', 'every',
        'exist', 'field', 'first', 'five', 'flame', 'floor', 'focus', 'force',
        'found', 'front', 'ghost', 'given', 'going', 'grant', 'great', 'green',
        'group', 'guide', 'happy', 'heart', 'heavy', 'hello', 'house', 'human', 'image',
        'world', 'would', 'wrote',
        'input', 'large', 'later', 'laugh', 'layer', 'learn', 'leave', 'level',
        'light', 'limit', 'local', 'login', 'lower', 'lucky', 'major', 'match',
        'might', 'model', 'money', 'month', 'mouse', 'music', 'night', 'noise',
        'north', 'noted', 'nurse', 'often', 'order', 'other', 'outer', 'owner',
        'panel', 'paper', 'party', 'phone', 'pilot', 'place', 'plain', 'plan',
        'plant', 'point', 'power', 'press', 'price', 'print', 'prior', 'prize',
        'proof', 'prove', 'proxy', 'queen', 'query', 'quest', 'queue', 'quick',
        'quiet', 'quote', 'radar', 'radio', 'raise', 'range', 'rapid', 'reach',
        'ready', 'reset', 'right', 'river', 'robot', 'rough', 'round', 'route',
        'ruler', 'saint', 'scale', 'scene', 'score', 'sense', 'serve', 'seven',
        'shall', 'shape', 'share', 'shelf', 'shift', 'shirt', 'shoot', 'shore',
        'short', 'shout', 'sight', 'since', 'sixth', 'sixty', 'skill', 'sleep',
        'slice', 'slide', 'smart', 'smile', 'smoke', 'solid', 'solve', 'sorry',
        'sound', 'south', 'space', 'speak', 'speed', 'spend', 'split', 'sport',
        'stack', 'stage', 'stand', 'start', 'state', 'steal', 'steam', 'steel',
        'stick', 'still', 'stock', 'store', 'storm', 'story', 'study', 'style',
        'sugar', 'super', 'swift', 'sword', 'table', 'teach', 'their', 'there',
        'these', 'thick', 'thing', 'think', 'third', 'three', 'throw', 'tiger',
        'timer', 'tired', 'today', 'token', 'topic', 'total', 'touch', 'tough',
        'track', 'trade', 'trail', 'train', 'treat', 'trial', 'trick', 'tried',
        'trust', 'truth', 'under', 'union', 'until', 'upper', 'urban', 'usage',
        'usual', 'valid', 'value', 'video', 'visit', 'voice', 'watch', 'water',
        'where', 'which', 'while', 'white', 'whole', 'whose', 'width', 'world',
        'worry', 'worst', 'worth', 'would', 'write', 'wrong', 'young',
    ]
