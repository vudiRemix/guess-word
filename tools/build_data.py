"""Собирает словарь и векторы для игры из модели RusVectores.

Модель: ruwikiruscorpora_upos_cbow_300_10_2021 (НКРЯ + Википедия, леммы с частями речи),
репозиторий NLPL, ID 220.

Использование:
    pip install numpy pymorphy3 pymorphy3-dicts-ru
    curl -LO http://vectors.nlpl.eu/repository/20/220.zip && unzip 220.zip model.txt
    python tools/build_data.py model.txt

Результат кладётся в data/:
    words.txt     — существительные (по одному в строке, по убыванию частоты;
                    «ё» сохраняется для показа, сравнение идёт без неё)
    vectors.bin   — PQ-коды, uint8 [len(words), 100]
    codebook.bin  — центроиды PQ, float32 [100, 256, 3]
    forms.json    — множественное число -> начальная форма (обе без «ё»)
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pymorphy3

VOCAB_SIZE = 30000
WORD_RE = re.compile(r"^[а-яё]{2,}$")
BAD_GRAMMEMES = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad", "Abbr", "Init", "Erro", "Dist", "Infr", "Slng", "Arch"}

# Product quantization: вектор режется на 100 кусков по 3 числа, каждый кусок
# заменяется номером ближайшего из 256 центроидов — 100 байт на слово.
SUBVECTORS = 100
CENTROIDS = 256
KMEANS_ITERATIONS = 25

OUT = Path(__file__).resolve().parent.parent / "data"


def noun_lemma(morph, word):
    """Начальная форма существительного (с «ё») или None, если word — не она."""
    if not morph.word_is_known(word):
        return None
    for p in morph.parse(word):
        tag = p.tag
        if (tag.POS == "NOUN" and p.normal_form.replace("ё", "е") == word and tag.case == "nomn"
                and not BAD_GRAMMEMES & set(tag.grammemes)):
            return p.normal_form
    return None


def read_nouns(path, morph):
    """Существительные модели в порядке убывания частоты (так их хранит gensim)."""
    lemmas, vectors, known = [], [], set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            token, rest = line.split(" ", 1)
            word, _, pos = token.rpartition("_")
            if pos != "NOUN" or not WORD_RE.match(word):
                continue
            key = word.replace("ё", "е")
            if key in known:
                continue
            lemma = noun_lemma(morph, key)
            if lemma is None:
                continue
            known.add(key)
            lemmas.append(lemma)
            vectors.append(np.array(rest.split(), dtype=np.float32))
            if len(lemmas) == VOCAB_SIZE:
                break
    return lemmas, np.stack(vectors)


def quantize(vectors, seed=0):
    rng = np.random.default_rng(seed)
    n, dim = vectors.shape
    subdim = dim // SUBVECTORS
    codebook = np.zeros((SUBVECTORS, CENTROIDS, subdim), dtype=np.float32)
    codes = np.zeros((n, SUBVECTORS), dtype=np.uint8)
    for j in range(SUBVECTORS):
        x = vectors[:, j * subdim:(j + 1) * subdim]
        centers = x[rng.choice(n, CENTROIDS, replace=False)].copy()
        for _ in range(KMEANS_ITERATIONS):
            dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels = dist.argmin(1)
            counts = np.bincount(labels, minlength=CENTROIDS)
            sums = np.stack([np.bincount(labels, weights=x[:, d], minlength=CENTROIDS) for d in range(subdim)], 1)
            filled = counts > 0
            centers[filled] = sums[filled] / counts[filled, None]
            # пустые кластеры переносим в случайные точки
            empty = ~filled
            centers[empty] = x[rng.integers(n, size=empty.sum())]
        codebook[j] = centers
        codes[:, j] = labels
    return codes, codebook


def main(path):
    morph = pymorphy3.MorphAnalyzer()
    lemmas, vectors = read_nouns(path, morph)

    # Нормируем и центрируем: у всех векторов есть общая составляющая,
    # из-за которой похожими кажутся просто частые слова.
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors -= vectors.mean(0)
    codes, codebook = quantize(vectors)

    keys = {w.replace("ё", "е") for w in lemmas}
    forms = {}
    for w in lemmas:
        plural = morph.parse(w)[0].inflect({"plur", "nomn"})
        if plural:
            form = plural.word.replace("ё", "е")
            if form not in keys:
                forms.setdefault(form, w.replace("ё", "е"))

    OUT.mkdir(exist_ok=True)
    (OUT / "words.txt").write_text("\n".join(lemmas) + "\n", encoding="utf-8")
    codes.tofile(OUT / "vectors.bin")
    codebook.tofile(OUT / "codebook.bin")
    (OUT / "forms.json").write_text(json.dumps(forms, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"words: {len(lemmas)}, forms: {len(forms)}")


if __name__ == "__main__":
    main(sys.argv[1])
