import { Model, encodeWord, decodeWord, normalize } from "./engine.js";
import { LEVELS } from "./levels.js";

const MODES = ["easy", "medium", "hard"];
const MODE_NAMES = { easy: "Лёгкий 🐥", medium: "Средний 🦊", hard: "Сложный 🦉" };
const MODE_ADJ = { easy: "лёгкое", medium: "среднее", hard: "сложное" };

// Новые слова появляются каждый день в 18:00 по Москве (15:00 UTC).
// Уровень №1 — 1 октября 2026.
const EPOCH = Date.UTC(2026, 9, 1, 15);
const DAY = 24 * 60 * 60 * 1000;
const GREEN = 100;
const YELLOW = 1000;
const FIRST_HINT = 300;
const CLOSEST = 500;

const $ = (sel) => document.querySelector(sel);
const els = {
  levels: $(".levels"),
  title: $("#game-title"),
  attempts: $("#attempts"),
  hints: $("#hints"),
  hintsInfo: $("#hints-info"),
  eye: $(".eye"),
  finish: $("#finish"),
  form: $("#guess-form"),
  input: $("#guess-input"),
  submit: $("#guess-form button"),
  message: $("#message"),
  last: $("#last"),
  history: $("#history"),
  loading: $("#loading"),
  dialog: $("#dialog"),
  dialogBody: $("#dialog-body"),
  toast: $("#toast"),
};

// ---------- хранилище ----------

const store = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem("gw:" + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem("gw:" + key, JSON.stringify(value));
    } catch {
      /* приватный режим и т.п. — играем без сохранения */
    }
  },
};

// ---------- утилиты ----------

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const plural = (n, [one, few, many]) => {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
};
const ATTEMPTS = ["попытку", "попытки", "попыток"];
const ATTEMPTS_NOM = ["попытка", "попытки", "попыток"];
const HINTS = ["подсказку", "подсказки", "подсказок"];

const todayNumber = () => Math.max(1, Math.floor((Date.now() - EPOCH) / DAY) + 1);
const nextReset = () => EPOCH + todayNumber() * DAY;

function seededShuffle(list, seedText) {
  let seed = [...seedText].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619), 2166136261) >>> 0;
  const random = () => {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = [...list];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function toast(text) {
  els.toast.textContent = text;
  els.toast.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (els.toast.hidden = true), 2200);
}

async function share(text) {
  if (navigator.share) {
    try {
      await navigator.share({ text });
      return;
    } catch (e) {
      if (e.name === "AbortError") return;
    }
  }
  try {
    await navigator.clipboard.writeText(text);
    toast("Скопировано!");
  } catch {
    prompt("Скопируйте текст:", text);
  }
}

const colorOf = (rank) => (rank <= GREEN ? "green" : rank <= YELLOW ? "yellow" : "red");

// ---------- состояние ----------

let model;
let secrets; // mode -> [индексы слов]
let game; // { mode, id, secret, key, url }
let state; // { guesses: [{ i, hint }], last, status }
let ranking;

function secretFor(mode, day) {
  const list = secrets[mode];
  return list[(day - 1) % list.length];
}

const pageUrl = (hash) => location.origin + location.pathname + hash;

function parseRoute() {
  const [, mode, id] = location.hash.split("/");
  if (mode === "custom" && id) {
    const secret = model.find(decodeWord(id));
    if (secret >= 0) return { mode, id, secret };
  }
  const level = MODES.includes(mode) ? mode : store.get("mode", "easy");
  let day = parseInt(id, 10);
  if (!(day >= 1 && day <= todayNumber())) day = todayNumber();
  return { mode: level, id: day, secret: secretFor(level, day) };
}

function openGame() {
  const route = parseRoute();
  const hash = `#/${route.mode}/${route.id}`;
  if (location.hash !== hash) history.replaceState(null, "", hash);
  if (route.mode !== "custom") store.set("mode", route.mode);

  game = { ...route, key: `game2:${route.mode}:${route.id}`, url: pageUrl(hash) };
  state = loadGame(game.key);
  ranking = model.ranking(game.secret);
  els.message.textContent = "";
  els.message.className = "message";
  render();
}

// В хранилище кладём сами слова, а не их номера: номера меняются при пересборке словаря.
function loadGame(key) {
  const saved = store.get(key, null);
  if (!saved) return { guesses: [], last: -1, status: "playing" };
  return {
    guesses: saved.guesses.map((g) => ({ i: model.find(g.w), hint: g.hint })).filter((g) => g.i >= 0),
    last: saved.last ? model.find(saved.last) : -1,
    status: saved.status,
  };
}

function saveGame() {
  store.set(game.key, {
    guesses: state.guesses.map((g) => ({ w: model.words[g.i], hint: g.hint })),
    last: state.last >= 0 ? model.words[state.last] : null,
    status: state.status,
  });
  if (state.status !== "playing" && game.mode !== "custom") {
    const results = store.get("results", {});
    results[`${game.mode}:${game.id}`] = {
      status: state.status,
      attempts: attempts(),
      hints: hints(),
    };
    store.set("results", results);
  }
}

const attempts = () => state.guesses.filter((g) => !g.hint).length;
const hints = () => state.guesses.filter((g) => g.hint).length;
const rankOf = (i) => ranking.rank[i];
const isToday = () => game.mode !== "custom" && game.id === todayNumber();

// ---------- отрисовка ----------

function rowHtml(i, { current = false, hint = false } = {}) {
  const rank = rankOf(i);
  const width = Math.max(2, 100 * (1 - Math.log(rank) / Math.log(model.maxRank)));
  const cls = ["row", colorOf(rank), current && "current", hint && "hint", rank === 1 && "win"].filter(Boolean).join(" ");
  return `<div class="${cls}">
    <div class="bar" style="width:${width.toFixed(1)}%"></div>
    <span class="word">${escapeHtml(model.words[i])}</span>
    <span class="rank">${rank}</span>
  </div>`;
}

function render() {
  for (const btn of els.levels.querySelectorAll("button")) {
    btn.classList.toggle("active", btn.dataset.level === game.mode);
  }
  els.title.textContent = game.mode === "custom" ? "Своё слово" : `Уровень №${game.id}`;
  els.attempts.textContent = attempts();
  els.hints.textContent = hints();
  els.hintsInfo.hidden = hints() === 0;

  const playing = state.status === "playing";
  els.input.disabled = els.submit.disabled = !playing;
  els.form.hidden = !playing;

  const last = state.guesses.find((g) => g.i === state.last);
  els.last.innerHTML = last && playing ? rowHtml(last.i, { current: true, hint: last.hint }) : "";
  els.history.innerHTML = [...state.guesses]
    .sort((a, b) => rankOf(a.i) - rankOf(b.i))
    .map((g) => rowHtml(g.i, { current: g.i === state.last && !playing, hint: g.hint }))
    .join("");

  renderFinish();
}

function renderFinish() {
  if (state.status === "playing") {
    els.finish.hidden = true;
    return;
  }
  const word = escapeHtml(model.words[game.secret]);
  const n = attempts();
  const h = hints();
  const won = state.status === "won";
  const hintText = h ? ` и ${h} ${plural(h, HINTS)}` : "";

  els.finish.hidden = false;
  els.finish.innerHTML = `
    <h2>${won ? (n <= 15 ? "НУ ТЫ КРУТОЙ!" : n <= 50 ? "С ПОБЕДОЙ!" : "ВОТ ЭТО ДА!") : "В другой раз!"}</h2>
    <div class="secret">Загаданное слово: <b>${word}</b></div>
    <p>${
      won
        ? `Ты справился за <b>${n}</b> ${plural(n, ATTEMPTS)}${hintText}`
        : `Ты сдался после <b>${n}</b> ${plural(n, ATTEMPTS)}${hintText}`
    }</p>
    <div class="actions">
      ${won ? `<button class="btn" data-action="share">Поделиться</button>` : ""}
      <button class="btn secondary" data-action="closest">Ближайшие слова</button>
      <button class="btn secondary" data-action="archive">Другой уровень</button>
      ${game.mode === "custom" ? `<button class="btn secondary" data-action="create">Загадать своё</button>` : ""}
    </div>
    ${isToday() ? `<div class="countdown">Новое ${MODE_ADJ[game.mode]} слово через: <b id="countdown"></b></div>` : ""}
  `;
  tick();
}

function tick() {
  const el = document.getElementById("countdown");
  if (!el) return;
  const left = nextReset() - Date.now();
  if (left <= 0 || !isToday()) {
    el.parentElement.innerHTML = `<button class="btn" data-action="today">Сыграть в новое слово</button>`;
    return;
  }
  const s = Math.floor(left / 1000);
  el.textContent = [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map((v) => String(v).padStart(2, "0")).join(":");
}

function showMessage(text, error = false) {
  els.message.textContent = text;
  els.message.className = "message" + (error ? " error" : "");
}

// ---------- ходы ----------

function addGuess(i, hint = false) {
  state.guesses.push({ i, hint });
  state.last = i;
  if (rankOf(i) === 1) state.status = "won";
  saveGame();
  render();
}

function onGuess(e) {
  e.preventDefault();
  const raw = els.input.value.trim();
  if (!raw || state.status !== "playing") return;
  els.input.value = "";

  if (!/^[а-яё-]+$/i.test(raw)) {
    showMessage("Слово может содержать только русские буквы", true);
    return;
  }
  const i = model.find(raw);
  if (i < 0) {
    showMessage(`Я не знаю слова «${raw}». Нужно существительное в начальной форме, например «собака».`, true);
    return;
  }
  if (state.guesses.some((g) => g.i === i)) {
    state.last = i;
    render();
    showMessage(`Слово «${model.words[i]}» уже было — ${rankOf(i)} место`);
    return;
  }
  showMessage(model.index.has(normalize(raw)) ? "" : `Засчитано как «${model.words[i]}»`);
  addGuess(i);
  if (state.status === "won") window.scrollTo({ top: 0, behavior: "smooth" });
}

function giveHint() {
  if (state.status !== "playing") return;
  const guessed = new Set(state.guesses.map((g) => g.i));
  const best = Math.min(Infinity, ...state.guesses.map((g) => rankOf(g.i)));
  let target = best === Infinity ? FIRST_HINT : Math.max(2, Math.floor(best / 2));
  // Ищем ближайшее ещё не названное слово: сначала ближе к загаданному, потом дальше.
  let i = -1;
  for (let r = target; r >= 2 && i < 0; r--) if (!guessed.has(ranking.order[r - 1])) i = ranking.order[r - 1];
  for (let r = target + 1; r <= ranking.order.length && i < 0; r++) if (!guessed.has(ranking.order[r - 1])) i = ranking.order[r - 1];
  if (i < 0) return;
  addGuess(i, true);
  showMessage(`Подсказка: «${model.words[i]}» — ${rankOf(i)} место`);
}

function giveUp() {
  if (state.status !== "playing") return;
  state.status = "lost";
  saveGame();
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function shareResult() {
  const counts = { green: 0, yellow: 0, red: 0 };
  for (const g of state.guesses) counts[colorOf(rankOf(g.i))]++;
  const n = attempts();
  const h = hints();
  const title = game.mode === "custom" ? "своё слово" : `${MODE_NAMES[game.mode]}, №${game.id}`;
  const text = [
    `Я отгадал слово в «Угадай слово» (${title}) за ${n} ${plural(n, ATTEMPTS)}${h ? `, ${h} ${plural(h, HINTS)}` : ""}`,
    `🟩 ${counts.green}  🟨 ${counts.yellow}  🟥 ${counts.red}`,
    `Сможешь быстрее? ${game.url}`,
  ].join("\n");
  share(text);
}

// ---------- окна ----------

function openDialog(html) {
  els.dialogBody.innerHTML = html;
  if (!els.dialog.open) els.dialog.showModal();
  els.dialog.scrollTop = 0;
}

function closeDialog() {
  if (els.dialog.open) els.dialog.close();
}

function showHelp() {
  const dog = model.find("собака");
  const example = (word) => {
    const i = model.find(word);
    const rank = model.ranking(dog).rank[i];
    const width = Math.max(2, 100 * (1 - Math.log(rank) / Math.log(model.maxRank)));
    return `<div class="row ${colorOf(rank)}"><div class="bar" style="width:${width}%"></div><span class="word">${word}</span><span class="rank">${rank}</span></div>`;
  };
  openDialog(`
    <h2>Как играть?</h2>
    <p>Я загадал слово. Попробуй его угадать! Попытки не ограничены.</p>
    <p>После каждого слова я скажу, насколько оно близко к загаданному, — смотри на его <b>место в рейтинге</b>.
      Загаданное слово стоит на 1-м месте. Чем меньше число, тем ближе ты к победе!</p>
    <p>Близость считается по смыслу и контексту: слова, которые часто встречаются рядом, близки друг к другу.</p>
    <h3>Пример: загадано «собака»</h3>
    <div class="legend">
      ${example("электричество")}
      ${example("друг")}
      ${example("животное")}
      ${example("собака")}
    </div>
    <p>🟩 — до ${GREEN}-го места, очень горячо<br />🟨 — до ${YELLOW}-го, тепло<br />🟥 — дальше, холодно</p>
    <h3>Что ещё</h3>
    <p>Вводи существительные в начальной форме: «кот», а не «котов».</p>
    <p>Новые слова трёх уровней сложности появляются каждый день в 18:00 по Москве. Прошлые уровни — в меню «Другой уровень».</p>
    <p>💡 Подсказка открывает слово ближе твоего лучшего. 👁 скрывает слова, если играешь на стриме.</p>
    <p>✏️ Загадай своё слово и отправь ссылку друзьям.</p>
    <p><button class="btn" data-action="close">Начать играть</button></p>
  `);
  store.set("seen", true);
}

function showMenu() {
  const playing = state && state.status === "playing";
  openDialog(`
    <h2>Меню</h2>
    <div class="menu">
      <button data-action="hint" ${playing ? "" : "disabled"}>💡 Подсказка</button>
      <button data-action="give-up" ${playing ? "" : "disabled"}>🏳️ Сдаться</button>
      <button data-action="archive">📅 Другой уровень</button>
      <button data-action="create">✏️ Загадать своё слово</button>
      <button data-action="stats">📊 Статистика</button>
      <button data-action="help">❓ Как играть</button>
    </div>
  `);
}

function confirmGiveUp() {
  openDialog(`
    <h2>Сдаться?</h2>
    <p>Загаданное слово будет показано, а уровень засчитается как проигранный.</p>
    <div class="menu">
      <button data-action="give-up-confirm">🏳️ Да, показать слово</button>
      <button data-action="close">Продолжить играть</button>
    </div>
  `);
}

function showArchive() {
  const results = store.get("results", {});
  const today = todayNumber();
  const sections = MODES.map((mode) => {
    const days = [];
    for (let d = today; d >= 1; d--) {
      const r = results[`${mode}:${d}`];
      const cls = [r && (r.status === "won" ? "done" : "lost"), game.mode === mode && game.id === d && "current"]
        .filter(Boolean)
        .join(" ");
      days.push(`<button class="${cls}" data-go="#/${mode}/${d}">${r && r.status === "won" ? "✓" : d}</button>`);
    }
    return `<h3>${MODE_NAMES[mode]}</h3><div class="archive">${days.join("")}</div>`;
  });
  openDialog(`<h2>Уровни</h2><p>Когда угадаешь слово — появится галочка.</p>${sections.join("")}`);
}

function showStats() {
  const results = store.get("results", {});
  const blocks = MODES.map((mode) => {
    const list = Object.entries(results)
      .filter(([k]) => k.startsWith(mode + ":"))
      .map(([, v]) => v);
    const wins = list.filter((r) => r.status === "won");
    const avg = wins.length ? Math.round(wins.reduce((s, r) => s + r.attempts, 0) / wins.length) : "—";
    const best = wins.length ? Math.min(...wins.map((r) => r.attempts)) : "—";
    return `<h3>${MODE_NAMES[mode]}</h3>
      <div class="stats">
        <div><b>${wins.length}/${list.length}</b><small>отгадано</small></div>
        <div><b>${avg}</b><small>${plural(typeof avg === "number" ? avg : 5, ATTEMPTS_NOM)} в среднем</small></div>
        <div><b>${best}</b><small>лучший результат</small></div>
      </div>`;
  });

  // Серия: подряд идущие дни, в которые отгадан хотя бы один ежедневный уровень.
  const wonDays = new Set(
    Object.entries(results)
      .filter(([, v]) => v.status === "won")
      .map(([k]) => +k.split(":")[1]),
  );
  let day = todayNumber();
  if (!wonDays.has(day)) day--;
  let streak = 0;
  while (wonDays.has(day)) {
    streak++;
    day--;
  }

  openDialog(`
    <h2>Статистика</h2>
    <div class="stats" style="grid-template-columns:1fr">
      <div><b>🔥 ${streak}</b><small>${plural(streak, ["день", "дня", "дней"])} подряд</small></div>
    </div>
    ${blocks.join("")}
  `);
}

function showCreate(prefill = "") {
  openDialog(`
    <h2>Загадай своё слово</h2>
    <p>Придумай слово и отправь ссылку друзьям — посмотрим, кто отгадает быстрее!</p>
    <form id="create-form">
      <label class="field">Слово
        <input id="create-input" type="text" placeholder="Например, «кит»" value="${escapeHtml(prefill)}" autocomplete="off" />
      </label>
      <p id="create-message" class="message"></p>
      <button class="btn" type="submit">Создать</button>
    </form>
    <div id="create-result"></div>
  `);
  const input = $("#create-input");
  input.focus();
  $("#create-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const msg = $("#create-message");
    const raw = input.value.trim();
    const i = /^[а-яё-]+$/i.test(raw) ? model.find(raw) : -1;
    if (i < 0) {
      msg.className = "message error";
      msg.textContent = raw ? "Такого слова нет в словаре. Нужно существительное, только русские буквы." : "Введите слово";
      $("#create-result").innerHTML = "";
      return;
    }
    msg.className = "message";
    msg.textContent = `Загадано: «${model.words[i]}»`;
    const hash = `#/custom/${encodeWord(model.words[i])}`;
    const url = pageUrl(hash);
    $("#create-result").innerHTML = `
      <p class="share-link">${escapeHtml(url)}</p>
      <div class="menu">
        <button data-action="share-custom" data-url="${escapeHtml(url)}">📤 Отправить друзьям</button>
        <button data-go="${hash}">▶️ Играть самому</button>
      </div>`;
  });
}

function showClosest() {
  const guessed = new Set(state.guesses.map((g) => g.i));
  const rows = [];
  for (let k = 0; k < CLOSEST; k++) {
    const i = ranking.order[k];
    rows.push(rowHtml(i, { current: guessed.has(i) }));
  }
  openDialog(`
    <h2>Ближайшие слова</h2>
    <p>${CLOSEST} самых близких к «${escapeHtml(model.words[game.secret])}» слов. Названные тобой — в рамке.</p>
    <div class="list closest">${rows.join("")}</div>
  `);
}

// ---------- события ----------

const actions = {
  help: showHelp,
  menu: showMenu,
  close: closeDialog,
  hint() {
    closeDialog();
    giveHint();
  },
  "give-up": confirmGiveUp,
  "give-up-confirm"() {
    closeDialog();
    giveUp();
  },
  archive: showArchive,
  create: () => showCreate(),
  stats: showStats,
  closest: showClosest,
  share: shareResult,
  "share-custom"(btn) {
    share(`Я загадал слово! Сможешь отгадать? ${btn.dataset.url}`);
  },
  today() {
    location.hash = `#/${game.mode}/${todayNumber()}`;
  },
  "toggle-hidden"() {
    const on = !document.body.classList.contains("hidden-words");
    document.body.classList.toggle("hidden-words", on);
    els.eye.classList.toggle("on", on);
    store.set("hidden", on);
  },
};

document.addEventListener("click", (e) => {
  const go = e.target.closest("[data-go]");
  if (go) {
    closeDialog();
    location.hash = go.dataset.go;
    return;
  }
  const level = e.target.closest("[data-level]");
  if (level && model) {
    location.hash = `#/${level.dataset.level}/${todayNumber()}`;
    return;
  }
  const btn = e.target.closest("[data-action]");
  if (!btn) return;
  const action = actions[btn.dataset.action];
  if (!action) return;
  if (!model && btn.dataset.action !== "close") return;
  action(btn);
});

els.dialog.addEventListener("click", (e) => {
  if (e.target === els.dialog) closeDialog();
});

els.form.addEventListener("submit", onGuess);
window.addEventListener("hashchange", () => model && openGame());
setInterval(tick, 1000);

// ---------- запуск ----------

if (store.get("hidden", false)) {
  document.body.classList.add("hidden-words");
  els.eye.classList.add("on");
}

try {
  model = await Model.load();
  secrets = Object.fromEntries(
    MODES.map((mode) => {
      const words = seededShuffle(LEVELS[mode].split(/\s+/).filter(Boolean), mode);
      const ids = words.map((w) => model.find(w));
      ids.forEach((id, k) => id < 0 && console.warn("Нет в словаре:", words[k]));
      return [mode, ids.filter((id) => id >= 0)];
    }),
  );
  els.loading.hidden = true;
  openGame();
  if (!store.get("seen", false)) showHelp();
  else els.input.focus();
} catch (err) {
  console.error(err);
  els.loading.innerHTML = `<p>Не удалось загрузить игру 😔<br />${escapeHtml(err.message)}</p>`;
}
