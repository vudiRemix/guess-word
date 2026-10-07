// Семантическая близость слов на векторах RusVectores (см. tools/build_data.py).
// Векторы хранятся в сжатом виде (product quantization): у каждого слова
// 100 байт — номера центроидов, сами центроиды лежат в codebook.bin.

const SUBVECTORS = 100;
const CENTROIDS = 256;
const SUBDIM = 3;

export const normalize = (s) => s.trim().toLowerCase().replace(/ё/g, "е");

export class Model {
  static async load(base = "data/") {
    const get = async (name, kind) => {
      const res = await fetch(base + name);
      if (!res.ok) throw new Error(`Не удалось загрузить ${name}`);
      return res[kind]();
    };
    const [words, codes, codebook, forms] = await Promise.all([
      get("words.txt", "text"),
      get("vectors.bin", "arrayBuffer"),
      get("codebook.bin", "arrayBuffer"),
      get("forms.json", "json"),
    ]);
    return new Model(words.trim().split("\n"), new Uint8Array(codes), new Float32Array(codebook), forms);
  }

  constructor(words, codes, codebook, forms) {
    this.words = words;
    this.codes = codes;
    this.codebook = codebook;
    this.forms = forms;
    this.index = new Map(words.map((w, i) => [normalize(w), i]));

    const centroidNorms = new Float32Array(SUBVECTORS * CENTROIDS);
    for (let k = 0; k < SUBVECTORS * CENTROIDS; k++) {
      let sum = 0;
      for (let d = 0; d < SUBDIM; d++) sum += codebook[k * SUBDIM + d] ** 2;
      centroidNorms[k] = sum;
    }
    this.norms = new Float32Array(words.length);
    for (let i = 0; i < words.length; i++) {
      let sum = 0;
      for (let j = 0; j < SUBVECTORS; j++) sum += centroidNorms[j * CENTROIDS + codes[i * SUBVECTORS + j]];
      this.norms[i] = Math.sqrt(sum) || 1;
    }
    this.cache = new Map();
  }

  get size() {
    return this.words.length;
  }

  // Индекс слова в словаре или -1. Понимает «ё» и множественное число.
  find(input) {
    const key = normalize(input);
    if (this.index.has(key)) return this.index.get(key);
    const lemma = this.forms[key];
    return lemma !== undefined && this.index.has(lemma) ? this.index.get(lemma) : -1;
  }

  // Рейтинг всех слов относительно загаданного: rank[i] — место слова i (1 — само слово),
  // order[k] — слово на месте k + 1.
  ranking(secret) {
    if (this.cache.has(secret)) return this.cache.get(secret);
    const { codes, codebook, norms } = this;
    const n = this.size;

    // Скалярное произведение каждого центроида с соответствующим куском загаданного вектора.
    const table = new Float32Array(SUBVECTORS * CENTROIDS);
    for (let j = 0; j < SUBVECTORS; j++) {
      const own = (j * CENTROIDS + codes[secret * SUBVECTORS + j]) * SUBDIM;
      for (let c = 0; c < CENTROIDS; c++) {
        const k = (j * CENTROIDS + c) * SUBDIM;
        let dot = 0;
        for (let d = 0; d < SUBDIM; d++) dot += codebook[k + d] * codebook[own + d];
        table[j * CENTROIDS + c] = dot;
      }
    }

    const score = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = 0; j < SUBVECTORS; j++) dot += table[j * CENTROIDS + codes[i * SUBVECTORS + j]];
      score[i] = dot / norms[i];
    }
    score[secret] = Infinity;

    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => score[b] - score[a]);
    const rank = new Uint32Array(n);
    for (let k = 0; k < n; k++) rank[order[k]] = k + 1;

    const result = { rank, order };
    this.cache.set(secret, result);
    return result;
  }
}

// Код своего слова для ссылки: слово не должно читаться прямо из адреса.
// Кодируется само слово, а не его номер, чтобы ссылки переживали пересборку словаря.
const KEY = [0x5b, 0xd1, 0xe9, 0x95, 0x3c, 0x7a];

export const encodeWord = (word) => {
  const bytes = new TextEncoder().encode(normalize(word)).map((b, k) => b ^ KEY[k % KEY.length]);
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
};

export const decodeWord = (code) => {
  try {
    const binary = atob(code.replace(/-/g, "+").replace(/_/g, "/"));
    const bytes = Uint8Array.from(binary, (c, k) => c.charCodeAt(0) ^ KEY[k % KEY.length]);
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return "";
  }
};
