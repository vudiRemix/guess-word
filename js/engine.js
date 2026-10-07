// Семантическая близость слов (векторы собирает tools/build_data.py).
// Векторы хранятся в сжатом виде (product quantization): у каждого слова
// по байту на кусок вектора — номер центроида, сами центроиды лежат в codebook.bin.
// Размеры описаны в meta.json.

export const normalize = (s) => s.trim().toLowerCase().replace(/ё/g, "е");

export class Model {
  static async load(base = "data/") {
    const get = async (name, kind) => {
      const res = await fetch(base + name);
      if (!res.ok) throw new Error(`Не удалось загрузить ${name}`);
      return res[kind]();
    };
    const [words, codes, codebook, forms, meta] = await Promise.all([
      get("words.txt", "text"),
      get("vectors.bin", "arrayBuffer"),
      get("codebook.bin", "arrayBuffer"),
      get("forms.json", "json"),
      get("meta.json", "json"),
    ]);
    return new Model(words.trim().split("\n"), new Uint8Array(codes), new Float32Array(codebook), forms, meta);
  }

  constructor(words, codes, codebook, forms, meta) {
    this.words = words;
    this.codes = codes;
    this.codebook = codebook;
    this.forms = forms;
    this.sub = meta.subvectors;
    this.centroids = meta.centroids;
    this.subdim = meta.subdim;
    // В рейтинге участвуют только первые (самые частые) слова, см. tools/build_data.py.
    this.ranked = Math.min(meta.ranked, words.length);
    this.index = new Map(words.map((w, i) => [normalize(w), i]));

    const { sub, centroids, subdim } = this;
    const centroidNorms = new Float32Array(sub * centroids);
    for (let k = 0; k < sub * centroids; k++) {
      let sum = 0;
      for (let d = 0; d < subdim; d++) sum += codebook[k * subdim + d] ** 2;
      centroidNorms[k] = sum;
    }
    this.norms = new Float32Array(words.length);
    for (let i = 0; i < words.length; i++) {
      let sum = 0;
      for (let j = 0; j < sub; j++) sum += centroidNorms[j * centroids + codes[i * sub + j]];
      this.norms[i] = Math.sqrt(sum) || 1;
    }
    this.cache = new Map();
  }

  get size() {
    return this.words.length;
  }

  // Самое большое возможное место: загаданное слово плюс все слова рейтинга.
  get maxRank() {
    return this.ranked + 1;
  }

  // Индекс слова в словаре или -1. Понимает «ё» и множественное число.
  find(input) {
    const key = normalize(input);
    if (this.index.has(key)) return this.index.get(key);
    const lemma = this.forms[key];
    return lemma !== undefined && this.index.has(lemma) ? this.index.get(lemma) : -1;
  }

  // rank[i] — место слова i (1 — загаданное), order[k] — слово на месте k + 1.
  // Место — это 1 + число слов рейтинга, которые ближе к загаданному.
  ranking(secret) {
    if (this.cache.has(secret)) return this.cache.get(secret);
    const { codes, codebook, norms, sub, centroids, subdim } = this;
    const n = this.size;

    // Скалярное произведение каждого центроида с соответствующим куском загаданного вектора.
    const table = new Float32Array(sub * centroids);
    for (let j = 0; j < sub; j++) {
      const own = (j * centroids + codes[secret * sub + j]) * subdim;
      for (let c = 0; c < centroids; c++) {
        const k = (j * centroids + c) * subdim;
        let dot = 0;
        for (let d = 0; d < subdim; d++) dot += codebook[k + d] * codebook[own + d];
        table[j * centroids + c] = dot;
      }
    }

    const score = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = 0; j < sub; j++) dot += table[j * centroids + codes[i * sub + j]];
      score[i] = dot / norms[i];
    }

    const others = [];
    for (let i = 0; i < this.ranked; i++) if (i !== secret) others.push(i);
    others.sort((a, b) => score[b] - score[a]);
    const order = Uint32Array.from([secret, ...others]);

    // Для каждого слова — бинарный поиск среди отсортированных слов рейтинга.
    const sorted = Float32Array.from(others, (i) => score[i]);
    const rank = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let lo = 0;
      let hi = sorted.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid] > score[i]) lo = mid + 1;
        else hi = mid;
      }
      rank[i] = lo + 2;
    }
    rank[secret] = 1;

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
