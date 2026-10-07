"""Собирает словарь и векторы для игры из модели navec.

Использование:
    pip install navec pymorphy3 pymorphy3-dicts-ru
    curl -LO https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
    python tools/build_data.py navec_hudlit_v1_12B_500K_300d_100q.tar

Результат кладётся в data/:
    words.txt     — существительные (по одному в строке, по убыванию частоты;
                    «ё» сохраняется для показа, сравнение идёт без неё)
    vectors.bin   — PQ-коды navec, uint8 [len(words), 100]
    codebook.bin  — центроиды PQ, float32 [100, 256, 3]
    forms.json    — множественное число -> начальная форма (обе без «ё»)
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pymorphy3
from navec import Navec

VOCAB_SIZE = 30000
WORD_RE = re.compile(r"^[а-яё]{2,}$")
NAME_GRAMMEMES = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad"}
BAD_GRAMMEMES = NAME_GRAMMEMES | {"Abbr", "Init", "Erro", "Dist", "Infr", "Slng", "Arch"}
# navec обучен на текстах в нижнем регистре, поэтому у слов-омонимов имён
# («ник», «боб», «генри») вектор описывает имя. Такие слова выкидываем,
# но обычные существительные, которые игроки наверняка введут, оставляем.
NAME_NEIGHBOURS = 10
NAME_SHARE = 0.4
KEEP_ANYWAY = set("""
    кит лев роза мак марс лилия юпитер султан джин амур астра лавр лира аврора
    вена марка катюша кантор
""".split())

OUT = Path(__file__).resolve().parent.parent / "data"


def is_lemma_noun(morph, word):
    if not morph.word_is_known(word):
        return False
    parses = morph.parse(word)
    # Маловероятные разборы не считаем, если только самый вероятный — не та же
    # начальная форма существительного (у «кит» он, например, помечен как имя).
    top = parses[0]
    lenient = top.tag.POS == "NOUN" and top.normal_form == word
    for p in parses:
        if p.score < 0.05 and not lenient:
            break
        tag = p.tag
        if (tag.POS == "NOUN" and p.normal_form == word and tag.case == "nomn"
                and not BAD_GRAMMEMES & set(tag.grammemes)):
            return True
    return False


def is_name(morph, word):
    return bool(NAME_GRAMMEMES & set(morph.parse(word)[0].tag.grammemes))


def name_dominated(navec, morph, order, candidates):
    """Слова из candidates, у которых среди ближайших соседей много имён."""
    pool = [i for i in order if WORD_RE.match(navec.vocab.words[i])][:100000]
    pq = navec.pq
    vectors = pq.codes[np.arange(pq.qdim)[None, :], pq.indexes[pool]].reshape(len(pool), pq.dim)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    position = {navec.vocab.words[i]: k for k, i in enumerate(pool)}

    result = set()
    for word in candidates:
        if word not in position:
            continue
        neighbours = np.argsort(-(vectors @ vectors[position[word]]))[1:NAME_NEIGHBOURS + 1]
        names = sum(is_name(morph, navec.vocab.words[pool[k]]) for k in neighbours)
        if names >= NAME_SHARE * NAME_NEIGHBOURS:
            result.add(word)
    return result


def main(path):
    navec = Navec.load(path)
    morph = pymorphy3.MorphAnalyzer()
    words, counts = navec.vocab.words, navec.vocab.counts

    order = sorted(range(len(words)), key=lambda i: -counts[i])
    candidates = [
        words[i] for i in order[:200000]
        if WORD_RE.match(words[i]) and is_name(morph, words[i]) and is_lemma_noun(morph, words[i])
    ]
    excluded = name_dominated(navec, morph, order, candidates) - KEEP_ANYWAY

    # В игре «ё» и «е» не различаются: из пары вариантов берём более частый.
    chosen, vocab, known = [], [], set()
    for i in order:
        w = words[i]
        key = w.replace("ё", "е")
        if WORD_RE.match(w) and key not in known and w not in excluded and is_lemma_noun(morph, w):
            chosen.append(i)
            vocab.append(w)
            known.add(key)
            if len(chosen) == VOCAB_SIZE:
                break

    forms = {}
    for i, w in zip(chosen, vocab):
        plural = morph.parse(words[i])[0].inflect({"plur", "nomn"})
        if plural:
            form = plural.word.replace("ё", "е")
            if form not in known:
                forms.setdefault(form, w.replace("ё", "е"))

    OUT.mkdir(exist_ok=True)
    (OUT / "words.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    navec.pq.indexes[chosen].astype(np.uint8).tofile(OUT / "vectors.bin")
    navec.pq.codes.astype(np.float32).tofile(OUT / "codebook.bin")
    (OUT / "forms.json").write_text(json.dumps(forms, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"words: {len(vocab)}, forms: {len(forms)}, excluded as names: {len(excluded)}")


if __name__ == "__main__":
    main(sys.argv[1])
