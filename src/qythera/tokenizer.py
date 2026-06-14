import re
import json
import math
import hashlib
import collections
from functools import lru_cache


class Vocabulary:
    """Character-level vocabulary with special tokens and byte fallback."""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_freqs = collections.Counter()
        self.special_tokens = {}
        self._init_base_vocab()

    def _init_base_vocab(self):
        base_tokens = list(range(256))
        for i, t in enumerate(base_tokens):
            self.token_to_id[f"<{i}>"] = i
            self.id_to_token[i] = f"<{i}>"
        for i in range(256):
            char = chr(i)
            if char not in self.token_to_id:
                tid = len(self.token_to_id)
                self.token_to_id[char] = tid
                self.id_to_token[tid] = char

    def add_token(self, token):
        if token not in self.token_to_id:
            tid = len(self.token_to_id)
            self.token_to_id[token] = tid
            self.id_to_token[tid] = token
        return self.token_to_id[token]

    def add_special_token(self, token, fixed_id=None):
        if fixed_id is not None:
            self.special_tokens[token] = fixed_id
            self.token_to_id[token] = fixed_id
            self.id_to_token[fixed_id] = token
        else:
            self.special_tokens[token] = len(self.token_to_id)
            self.add_token(token)

    def get_id(self, token):
        return self.token_to_id.get(token, self.token_to_id.get(f"<{ord(token)}>", -1))

    def get_token(self, tid):
        return self.id_to_token.get(tid, "<unk>")

    def update_freqs(self, tokens):
        self.token_freqs.update(tokens)

    def vocab_size(self):
        return len(self.token_to_id)

    def __len__(self):
        return self.vocab_size()

    def __contains__(self, token):
        return token in self.token_to_id


class BPETokenizer:
    """Byte-level BPE tokenizer with GPT-4 style byte mapping."""

    def __init__(self):
        self.byte_encoder = {}
        self.byte_decoder = {}
        self._build_byte_encoder()
        self.merges = []
        self.vocab = {}
        self.vocab_r = {}
        self.special_tokens = {}
        self._encode_cache = None
        self._decode_cache = None

    def _build_byte_encoder(self):
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        self.byte_encoder = dict(zip(bs, [chr(c) for c in cs]))
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def _bytes_to_unicode(self, b):
        return ''.join(self.byte_encoder.get(i, chr(i)) for i in b)

    def _unicode_to_bytes(self, s):
        return bytes([self.byte_decoder.get(c, ord(c)) for c in s])

    def _get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def _stats(self, word_freqs):
        pairs = collections.Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge(self, pair, word_freqs):
        bigram = pair
        new_word_freqs = {}
        repl = bigram[0] + bigram[1]
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(repl)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, corpus, vocab_size=256+5000):
        tokens_list = []
        for ch in corpus:
            tokens_list.append(self._bytes_to_unicode(ch.encode('utf-8')))
        base_vocab = set(tokens_list)
        word_freqs = collections.Counter()
        for t in tokens_list:
            word_freqs[t] += 1
        num_merges = vocab_size - 256
        merges = []
        vocab = {t: i for i, t in enumerate(base_vocab)}
        for i in range(num_merges):
            pairs = self._stats(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges.append(best)
            word_freqs = self._merge(best, word_freqs)
            merged = best[0] + best[1]
            vocab[merged] = 256 + i
        self.merges = merges
        self.vocab = {k: 256 + i for i, k in enumerate(base_vocab)}
        self.vocab.update({b[0] + b[1]: 256 + len(base_vocab) + i for i, b in enumerate(merges)})
        self.vocab_r = {v: k for k, v in self.vocab.items()}
        self._setup_cache()

    def _encode_inner(self, text):
        tokens = []
        for ch in text:
            bs = ch.encode('utf-8')
            tokens.append(self._bytes_to_unicode(bs))
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = pair[0] + pair[1]
                    tokens.pop(i + 1)
                else:
                    i += 1
        ids = [self.vocab.get(t, 0) for t in tokens]
        return ids

    def _decode_inner(self, ids):
        tokens = []
        for tid in ids:
            tokens.append(self.vocab_r.get(tid, ''))
        full_text = ''.join(tokens)
        bytes_list = []
        for ch in full_text:
            if ch in self.byte_decoder:
                bytes_list.append(self.byte_decoder[ch])
            else:
                bytes_list.append(ord(ch))
        return bytes(bytes_list).decode('utf-8', errors='replace')

    def _setup_cache(self):
        self._encode_cache = lru_cache(maxsize=2**16)(self._encode_inner)
        self._decode_cache = lru_cache(maxsize=2**16)(self._decode_inner)

    def _encode_bpe_dropout(self, text, p=0.1):
        import random
        tokens = []
        for ch in text:
            bs = ch.encode('utf-8')
            tokens.append(self._bytes_to_unicode(bs))
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    if random.random() >= p:
                        tokens[i] = pair[0] + pair[1]
                        tokens.pop(i + 1)
                    else:
                        i += 1
                else:
                    i += 1
        return [self.vocab.get(t, 0) for t in tokens]

    def _sample_segmentation(self, text, temperature=1.0):
        import random
        tokens = []
        for ch in text:
            bs = ch.encode('utf-8')
            tokens.append(self._bytes_to_unicode(bs))
        if len(tokens) <= 1:
            return tokens
        merge_rank = {pair: idx for idx, pair in enumerate(self.merges)}
        for _ in range(len(self.merges)):
            candidates = []
            i = 0
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i + 1])
                if pair in merge_rank:
                    rank = merge_rank[pair]
                    weight = 1.0 / (rank + 1)
                    candidates.append((i, pair, weight))
                i += 1
            if not candidates:
                break
            if temperature != 1.0:
                adjusted = [(pos, pair, w ** (1.0 / temperature)) for pos, pair, w in candidates]
            else:
                adjusted = candidates
            total = sum(w for _, _, w in adjusted)
            probs = [w / total for _, _, w in adjusted]
            idx = random.choices(range(len(adjusted)), weights=probs, k=1)[0]
            pos, pair, _ = adjusted[idx]
            tokens[pos] = pair[0] + pair[1]
            tokens.pop(pos + 1)
        return tokens

    def encode(self, text, dropout=None, temperature=None):
        if dropout is not None and 0 <= dropout < 1:
            return self._encode_bpe_dropout(text, p=dropout)
        if temperature is not None and temperature > 0:
            tokens = self._sample_segmentation(text, temperature=temperature)
            return [self.vocab.get(t, 0) for t in tokens]
        if self._encode_cache is None:
            self._setup_cache()
        return self._encode_cache(text)

    def encode_sample(self, text, temperature=1.0):
        tokens = self._sample_segmentation(text, temperature=temperature)
        return [self.vocab.get(t, 0) for t in tokens]

    def decode(self, ids):
        if self._decode_cache is None:
            self._setup_cache()
        return self._decode_cache(tuple(ids))

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch_ids):
        return [self.decode(ids) for ids in batch_ids]

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def add_special_tokens(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        for token in tokens:
            if token not in self.special_tokens:
                tid = self.vocab_size
                self.special_tokens[token] = tid
                self.vocab[token] = tid
                self.vocab_r[tid] = token

    def save(self, path):
        data = {
            'merges': [[a, b] for a, b in self.merges],
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.merges = [tuple(m) for m in data['merges']]
        self.vocab = data['vocab']
        self.vocab_r = {v: k for k, v in self.vocab.items()}
        self.special_tokens = data.get('special_tokens', {})
        self._setup_cache()

    def apply_chat_template(self, messages):
        parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return ''.join(parts)

    def encode_with_fim(self, prefix, suffix, mode='psm'):
        if mode == 'psm':
            text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        else:
            text = f"<|fim_suffix|>{prefix}<|fim_prefix|>{suffix}<|fim_middle|>"
        return self.encode(text)


class WordPieceTokenizer:
    """Greedy longest-match WordPiece tokenizer."""

    def __init__(self):
        self.vocab = {}
        self.vocab_r = {}
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"

    def build_vocab(self, corpus, vocab_size=10000):
        char_freqs = collections.Counter(corpus)
        for ch, freq in char_freqs.most_common(vocab_size - 5):
            self.vocab[ch] = len(self.vocab)
        for token in [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]:
            self.vocab[token] = len(self.vocab)
        self.vocab_r = {v: k for k, v in self.vocab.items()}

    def train(self, corpus, vocab_size=10000):
        self.build_vocab(corpus, vocab_size)

    def _tokenize_word(self, word):
        chars = list(word) + ['</w>']
        tokens = []
        i = 0
        while i < len(chars):
            found = False
            for j in range(len(chars), i, -1):
                subword = ''.join(chars[i:j])
                if j < len(chars):
                    subword = subword
                if subword in self.vocab:
                    tokens.append(subword)
                    i = j
                    found = True
                    break
            if not found:
                tokens.append(self.unk_token)
                i += 1
        return tokens

    def encode(self, text):
        tokens = text.split()
        result = []
        for word in tokens:
            word_tokens = self._tokenize_word(word)
            result.extend(word_tokens)
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in result]
        return ids

    def decode(self, ids):
        tokens = [self.vocab_r.get(tid, '') for tid in ids]
        text = ''
        for tok in tokens:
            if tok == '</w>':
                text += ' '
            else:
                text += tok
        return text.strip()

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch_ids):
        return [self.decode(ids) for ids in batch_ids]

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def add_special_tokens(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.vocab_r[self.vocab[token]] = token

    def save(self, path):
        data = {
            'vocab': self.vocab,
            'special_tokens': {
                'unk': self.unk_token,
                'cls': self.cls_token,
                'sep': self.sep_token,
                'pad': self.pad_token,
                'mask': self.mask_token,
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.vocab_r = {v: k for k, v in self.vocab.items()}
        st = data.get('special_tokens', {})
        self.unk_token = st.get('unk', '[UNK]')
        self.cls_token = st.get('cls', '[CLS]')
        self.sep_token = st.get('sep', '[SEP]')
        self.pad_token = st.get('pad', '[PAD]')
        self.mask_token = st.get('mask', '[MASK]')


class UnigramTokenizer:
    """Viterbi-based Unigram tokenizer with probability pruning."""

    def __init__(self):
        self.vocab = {}
        self.vocab_r = {}
        self.log_probs = {}
        self.special_tokens = {}

    def train(self, corpus, vocab_size=10000, iterations=10):
        char_freqs = collections.Counter()
        for line in corpus.split('\n'):
            for i in range(len(line)):
                for j in range(i + 1, min(i + 32, len(line) + 1)):
                    char_freqs[line[i:j]] += 1
        total = sum(char_freqs.values())
        self.log_probs = {k: math.log(v / total) for k, v in char_freqs.items()}
        self.vocab = {k: i for i, k in enumerate(sorted(char_freqs.keys(), key=lambda x: -char_freqs[x])[:vocab_size])}
        self.vocab_r = {v: k for k, v in self.vocab.items()}
        for _ in range(iterations):
            new_log_probs = {}
            for token, tid in self.vocab.items():
                if token in self.log_probs:
                    new_log_probs[token] = self.log_probs[token]
            self.log_probs = new_log_probs

    def _viterbi(self, text):
        n = len(text)
        best_score = [float('-inf')] * (n + 1)
        best_slice = [0] * (n + 1)
        best_score[0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(0, i - 32), i):
                sub = text[j:i]
                if sub in self.log_probs:
                    score = best_score[j] + self.log_probs[sub]
                    if score > best_score[i]:
                        best_score[i] = score
                        best_slice[i] = j
        tokens = []
        i = n
        while i > 0:
            j = best_slice[i]
            tokens.append(text[j:i])
            i = j
        tokens.reverse()
        return tokens

    def encode(self, text):
        tokens = self._viterbi(text)
        ids = [self.vocab.get(t, 0) for t in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.vocab_r.get(tid, '') for tid in ids]
        return ''.join(tokens)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch_ids):
        return [self.decode(ids) for ids in batch_ids]

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def add_special_tokens(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        for token in tokens:
            if token not in self.special_tokens:
                self.special_tokens[token] = len(self.vocab)
                self.vocab[token] = self.special_tokens[token]
                self.vocab_r[self.special_tokens[token]] = token
                self.log_probs[token] = 0.0

    def save(self, path):
        data = {
            'vocab': self.vocab,
            'log_probs': self.log_probs,
            'special_tokens': self.special_tokens,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.vocab_r = {v: k for k, v in self.vocab.items()}
        self.log_probs = data['log_probs']
        self.special_tokens = data.get('special_tokens', {})


class TiktokenTokenizer:
    """Regex pre-tokenization tokenizer matching cl100k base pattern."""

    def __init__(self):
        self.pattern = re.compile(
            r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\d]?+\w+|\d{1,3}| ?[^\s\w\d]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
        self.byte_encoder = {}
        self.byte_decoder = {}
        self._build_byte_encoder()
        self.vocab = {}
        self.vocab_r = {}
        self.special_tokens = {}

    def _build_byte_encoder(self):
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        self.byte_encoder = dict(zip(bs, [chr(c) for c in cs]))
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def _bytes_to_unicode(self, b):
        return ''.join(self.byte_encoder.get(i, chr(i)) for i in b)

    def train(self, corpus, vocab_size=50000):
        words = re.findall(self.pattern, corpus)
        word_freqs = collections.Counter()
        for w in words:
            encoded = []
            for ch in w:
                bs = ch.encode('utf-8')
                encoded.append(self._bytes_to_unicode(bs))
            word_freqs[tuple(encoded)] += 1
        base_vocab = list(self.byte_encoder.values())
        all_tokens = set()
        for word in word_freqs:
            for token in word:
                all_tokens.add(token)
        for i, token in enumerate(sorted(all_tokens)):
            if len(self.vocab) >= vocab_size:
                break
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        for i, b in enumerate(base_vocab):
            if b not in self.vocab:
                self.vocab[b] = len(self.vocab)
        self.vocab_r = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        words = re.findall(self.pattern, text)
        ids = []
        for w in words:
            for ch in w:
                bs = ch.encode('utf-8')
                token = self._bytes_to_unicode(bs)
                tid = self.vocab.get(token, 0)
                ids.append(tid)
        return ids

    def decode(self, ids):
        tokens = []
        for tid in ids:
            token = self.vocab_r.get(tid, '')
            tokens.append(token)
        full_text = ''.join(tokens)
        bytes_list = []
        for ch in full_text:
            if ch in self.byte_decoder:
                bytes_list.append(self.byte_decoder[ch])
            else:
                bytes_list.append(ord(ch))
        return bytes(bytes_list).decode('utf-8', errors='replace')

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch_ids):
        return [self.decode(ids) for ids in batch_ids]

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def add_special_tokens(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        for token in tokens:
            if token not in self.special_tokens:
                self.special_tokens[token] = len(self.vocab)
                self.vocab[token] = self.special_tokens[token]
                self.vocab_r[self.special_tokens[token]] = token

    def save(self, path):
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.vocab_r = {v: k for k, v in self.vocab.items()}
        self.special_tokens = data.get('special_tokens', {})
