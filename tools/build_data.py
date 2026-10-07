"""Собирает словарь и векторы для игры из двух моделей.

1. RusVectores ruwikiruscorpora_upos_cbow_300_10_2021 (НКРЯ + Википедия, леммы
   с частями речи; репозиторий NLPL, ID 220). Хорошо знает смысл слов, а имена
   собственные в ней размечены отдельно.
2. fastText cc.ru.300 (Common Crawl). Учитывает части слов, поэтому родственные
   слова близки: «коровка» — к «корове», хотя в текстах это чаще «божья коровка».

Векторы обеих моделей складываются в один (каждая даёт половину сходства):
на наборе RUSSE HJ корреляция с оценками людей 0,743 против 0,707 и 0,697
у моделей по отдельности.

Использование:
    pip install numpy pymorphy3 pymorphy3-dicts-ru
    curl -LO http://vectors.nlpl.eu/repository/20/220.zip && unzip 220.zip model.txt
    curl -LO https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz
    python tools/build_data.py model.txt cc.ru.300.vec.gz

Результат кладётся в data/:
    words.txt     — существительные (по одному в строке, по убыванию частоты;
                    «ё» сохраняется для показа, сравнение идёт без неё)
    vectors.bin   — PQ-коды, uint8 [len(words), SUBVECTORS]
    codebook.bin  — центроиды PQ, float32 [SUBVECTORS, CENTROIDS, SUBDIM]
    meta.json     — размеры PQ и сколько первых слов участвуют в рейтинге
    forms.json    — множественное число -> начальная форма (обе без «ё»)
"""

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pymorphy3

VOCAB_SIZE = 30000
# Место слова считается только среди самых частых слов: иначе верх рейтинга
# занимают редкие слова («листоед», «капустница»), которые никто не вводит.
# Вводить можно и остальные — они тоже получают место среди частых.
RANKED = 15000
WORD_RE = re.compile(r"^[а-яё]{2,}$")
BAD_GRAMMEMES = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad", "Abbr", "Init", "Erro", "Dist", "Infr", "Slng", "Arch"}

# Product quantization: вектор (600 чисел) режется на 200 кусков по 3 числа,
# каждый кусок заменяется номером ближайшего из 256 центроидов — 200 байт на слово.
SUBVECTORS = 200
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


def read_rusvectores(path, morph):
    """Существительные модели в порядке убывания частоты (так их хранит gensim)."""
    lemmas, vectors, known = [], {}, set()
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
            vectors[key] = np.array(rest.split(), dtype=np.float32)
    return lemmas, vectors


def read_fasttext(path, keys):
    vectors = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            token, rest = line.split(" ", 1)
            # только слова в нижнем регистре: «Ник» и «ник» в fastText разные слова
            if token != token.lower():
                continue
            key = token.replace("ё", "е")
            if key in keys and key not in vectors:
                vectors[key] = np.array(rest.split(), dtype=np.float32)
    return vectors


def prepare(vectors):
    """Нормируем и центрируем: у всех векторов есть общая составляющая,
    из-за которой похожими кажутся просто частые слова."""
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors -= vectors.mean(0)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


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
            dist = (x * x).sum(1)[:, None] - 2 * x @ centers.T + (centers * centers).sum(1)
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


def main(rusvectores_path, fasttext_path):
    morph = pymorphy3.MorphAnalyzer()
    lemmas, rv = read_rusvectores(rusvectores_path, morph)
    ft = read_fasttext(fasttext_path, set(rv))

    lemmas = [w for w in lemmas if w.replace("ё", "е") in ft][:VOCAB_SIZE]
    keys = [w.replace("ё", "е") for w in lemmas]
    half = np.sqrt(0.5)
    vectors = np.hstack([
        prepare(np.stack([rv[k] for k in keys])) * half,
        prepare(np.stack([ft[k] for k in keys])) * half,
    ]).astype(np.float32)
    codes, codebook = quantize(vectors)

    known = set(keys)
    forms = {}
    for w in lemmas:
        plural = morph.parse(w)[0].inflect({"plur", "nomn"})
        if plural:
            form = plural.word.replace("ё", "е")
            if form not in known:
                forms.setdefault(form, w.replace("ё", "е"))

    OUT.mkdir(exist_ok=True)
    (OUT / "words.txt").write_text("\n".join(lemmas) + "\n", encoding="utf-8")
    codes.tofile(OUT / "vectors.bin")
    codebook.tofile(OUT / "codebook.bin")
    meta = {"subvectors": SUBVECTORS, "centroids": CENTROIDS, "subdim": codebook.shape[2], "ranked": RANKED}
    (OUT / "meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    (OUT / "forms.json").write_text(json.dumps(forms, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"words: {len(lemmas)}, forms: {len(forms)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
