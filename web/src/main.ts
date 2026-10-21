import "./style.css";
import { registerSW } from "virtual:pwa-register";
import { GameController, Difficulty, Mode, EngineKind } from "./game-controller";
import { BLACK, Color, HAND_TYPES, Move, PieceType, WHITE, colorOf, typeOf } from "./engine/types";
import { NNUE_LEVELS } from "./player/nnue-config";
import type { RookRoute } from "./player/rook-guide";
import { Position, squareOf } from "./engine/position";
import { Language, MessageKey, detectLanguage, t } from "./i18n";
import { PeerSession, SessionSnapshot } from "./online/peer-session";
import { createRoomCode } from "./online/protocol";

type Screen = "home" | "setup-cpu" | "setup-online" | "game" | "about";
type Selection = { square: number | null; hand: PieceType | null };

const appElement = document.querySelector<HTMLDivElement>("#app");
if (!appElement) throw new Error("Missing #app root");
const app: HTMLDivElement = appElement;

const controller = new GameController();
let language = detectLanguage();
let setupEngine: EngineKind = "abp";
let screen: Screen = "home";
let flipped = false;
let sound = localStorage.getItem("alphasho.sound") !== "off";
let selection: Selection = { square: null, hand: null };
let promotionChoices: Move[] | null = null;
let peer: PeerSession | null = null;
let peerStatus: SessionSnapshot | null = null;
let onlineColor: Color | null = null;
let toast = "";
let updateAvailable = false;
let activateUpdate: ((reloadPage?: boolean) => Promise<void>) | null = null;

const pieceNames: Record<Language, Record<number, string>> = {
  ja: { 1: "歩", 2: "香", 3: "桂", 4: "銀", 5: "金", 6: "角", 7: "飛", 8: "王", 9: "と", 10: "杏", 11: "圭", 12: "全", 13: "馬", 14: "龍" },
  zh: { 1: "步", 2: "香", 3: "桂", 4: "银", 5: "金", 6: "角", 7: "飞", 8: "王", 9: "と", 10: "成香", 11: "成桂", 12: "成银", 13: "马", 14: "龙" },
  en: { 1: "P", 2: "L", 3: "N", 4: "S", 5: "G", 6: "B", 7: "R", 8: "K", 9: "+P", 10: "+L", 11: "+N", 12: "+S", 13: "+B", 14: "+R" }
};

function tr(key: MessageKey): string { return t(language, key); }
function setToast(message: string): void { toast = message; render(); window.setTimeout(() => { if (toast === message) { toast = ""; render(); } }, 2400); }

controller.onChange = render;

function topBar(): string {
  return `<header class="topbar"><button class="brand plain" data-action="home" aria-label="AlphaSho home">
    <span class="brand-mark">王</span><span>AlphaSho</span></button>
    <div class="top-actions">
      <button class="icon-button" data-action="sound" aria-label="${tr("sound")}" title="${tr("sound")}">${sound ? "♪" : "×"}</button>
      <select data-action="language" aria-label="Language"><option value="ja" ${language === "ja" ? "selected" : ""}>日本語</option>
        <option value="zh" ${language === "zh" ? "selected" : ""}>中文</option><option value="en" ${language === "en" ? "selected" : ""}>English</option></select>
    </div></header>`;
}

function homeView(): string {
  return `${topBar()}<main class="home-shell">
    <section class="hero"><div class="hero-kicker">BROWSER SHOGI</div><h1>Alpha<span>Sho</span></h1>
      <p>${tr("subtitle")}</p><div class="hero-piece" aria-hidden="true">王</div></section>
    <section class="mode-grid">
      <button class="mode-card cpu" data-action="setup-cpu"><span class="mode-icon">☖</span><span><strong>${tr("vsCpu")}</strong><small>${tr("easy")} · ${tr("normal")} · ${tr("hard")}</small></span><b>→</b></button>
      <button class="mode-card local" data-action="start-local"><span class="mode-icon">対</span><span><strong>${tr("local")}</strong><small>${tr("undo")} · ${tr("flip")}</small></span><b>→</b></button>
      <button class="mode-card online" data-action="setup-online"><span class="mode-icon">⌁</span><span><strong>${tr("online")}</strong><small>${tr("onlineNote")}</small></span><b>→</b></button>
    </section><button class="about-link plain" data-action="about">${tr("about")}</button>
  </main>`;
}

function segmented(name: string, options: Array<[string, string]>, selected: string): string {
  return `<div class="segmented">${options.map(([value, label]) => `<label><input type="radio" name="${name}" value="${value}" ${selected === value ? "checked" : ""}><span>${label}</span></label>`).join("")}</div>`;
}

function cpuSetup(): string {
  return `${topBar()}<main class="setup-shell"><button class="back plain" data-action="home">← ${tr("home")}</button>
    <div class="setup-card"><div class="setup-glyph">☖</div><h2>${tr("vsCpu")}</h2>
      <label class="field-label">${tr("chooseSide")}</label>${segmented("side", [["0", tr("black")], ["1", tr("white")], ["random", tr("random")]], "0")}
      <label class="field-label">${tr("engineLabel")}</label>${segmented("engine", [["abp", "ABP"], ["nnue", tr("nnueLabel")]], setupEngine)}
      <label class="field-label">${tr("chooseDifficulty")}</label>${setupEngine === "nnue" ? `<select name="nn-level" aria-label="${tr("nnLevel")}">${NNUE_LEVELS.map((l, i) => `<option value="${i+1}" ${i === 4 ? "selected" : ""}>${tr("nnLevel")} ${i+1} · ${l.milliseconds/1000}s</option>`).join("")}</select><p>${tr("nnueVersion")}</p>` : segmented("difficulty", [["easy", tr("easy")], ["normal", tr("normal")], ["hard", tr("hard")]], "normal")}
      <a href="${import.meta.env.BASE_URL}benchmark.html">${tr("benchmark")}</a>
      <button class="primary wide" data-action="start-cpu">${tr("start")}</button></div></main>`;
}

function onlineSetup(): string {
  const queryRoom = new URLSearchParams(location.search).get("room")?.toUpperCase() ?? "";
  return `${topBar()}<main class="setup-shell"><button class="back plain" data-action="home">← ${tr("home")}</button>
    <div class="setup-card"><div class="setup-glyph">⌁</div><h2>${tr("online")}</h2>
      <label class="field-label">${tr("hostSide")}</label>${segmented("host-side", [["0", tr("black")], ["1", tr("white")], ["random", tr("random")]], "random")}
      <button class="primary wide" data-action="create-room">${tr("createRoom")}</button><div class="or"><span></span><b>OR</b><span></span></div>
      <label class="field-label" for="room-code">${tr("roomCode")}</label><div class="join-row"><input id="room-code" maxlength="10" value="${escapeHtml(queryRoom)}" autocomplete="off" spellcheck="false">
      <button class="secondary" data-action="join-room">${tr("joinRoom")}</button></div><p class="fine-print">${tr("onlineNote")}</p></div></main>`;
}

function pieceAsset(piece: number): string {
  const color = colorOf(piece) === BLACK ? "black" : "white";
  const names: Record<number, string> = { 1: "pawn", 2: "lance", 3: "knight", 4: "silver", 5: "gold", 6: "bishop", 7: "rook", 8: "king",
    9: "prom_pawn", 10: "prom_lance", 11: "prom_knight", 12: "prom_silver", 13: "horse", 14: "dragon" };
  return `${import.meta.env.BASE_URL}pieces/${color}_${names[typeOf(piece)]}.png`;
}

function handView(position: Position, color: Color): string {
  const items = [...HAND_TYPES].reverse().filter((type) => (position.hands[color * 8 + type] ?? 0) > 0);
  return `<div class="hand ${color === BLACK ? "black-hand" : "white-hand"}" aria-label="${tr("hand")}">
    <span class="hand-owner">${color === BLACK ? tr("black") : tr("white")}</span>${items.length ? items.map((type) => {
      const selected = selection.hand === type && position.turn === color;
      const piece = color === BLACK ? type : -type;
      return `<button class="hand-piece ${selected ? "selected" : ""}" data-hand="${type}" ${position.turn !== color || !canInteract() ? "disabled" : ""}>
        <img src="${pieceAsset(piece)}" alt="${pieceNames[language][type]}"><b>×${position.hands[color * 8 + type]}</b></button>`;
    }).join("") : `<span class="empty-hand">—</span>`}</div>`;
}

function targetSquares(): Set<number> {
  if (selection.hand) return new Set(controller.legalMovesFrom(null, selection.hand).map((move) => move.to));
  if (selection.square !== null) return new Set(controller.legalMovesFrom(selection.square).map((move) => move.to));
  return new Set();
}

function boardView(position: Position): string {
  const targets = targetSquares();
  const last = controller.state.history.at(-1);
  let lastTo: number | null = null;
  if (last) { try { lastTo = position.parseUsi(last).to; } catch { lastTo = null; } }
  const squares: string[] = [];
  const files = flipped ? [...Array(9).keys()].reverse() : [...Array(9).keys()];
  const ranks = flipped ? [...Array(9).keys()].reverse() : [...Array(9).keys()];
  for (const rank of ranks) for (const file of files) {
    const square = squareOf(file, rank);
    const piece = position.board[square] ?? 0;
    const classes = ["square", targets.has(square) ? "target" : "", selection.square === square ? "selected" : "", lastTo === square ? "last" : ""].filter(Boolean).join(" ");
    squares.push(`<button class="${classes}" data-square="${square}" aria-label="${9 - file}${String.fromCharCode(97 + rank)}${piece ? ` ${pieceNames[language][typeOf(piece)]}` : ""}">
      ${piece ? `<img src="${pieceAsset(piece)}" alt="${pieceNames[language][typeOf(piece)]}">` : ""}${targets.has(square) ? `<i></i>` : ""}</button>`);
  }
  return `<div class="board-wrap"><div class="file-labels">${files.map((file) => `<span>${9 - file}</span>`).join("")}</div>
    <div class="board">${squares.join("")}</div><div class="rank-labels">${ranks.map((rank) => `<span>${String.fromCharCode(97 + rank)}</span>`).join("")}</div></div>`;
}

function canInteract(): boolean {
  if (!controller.canHumanMove()) return false;
  if (controller.state.mode === "online") return onlineColor !== null && controller.state.position.turn === onlineColor && peerStatus?.status === "connected";
  return true;
}

function outcomeText(): string | null {
  const outcome = controller.state.outcome;
  if (!outcome) return null;
  const winner = outcome.winner === null ? tr("draw") : outcome.winner === BLACK ? tr("winnerBlack") : tr("winnerWhite");
  const reasons: Record<string, MessageKey> = { checkmate: "checkmate", resignation: "resigned", repetition: "repetition", "perpetual-check": "perpetual", "entering-king": "enteringKing" };
  return `${winner} · ${tr(reasons[outcome.reason] ?? "checkmate")}`;
}

function statusText(): string {
  const ended = outcomeText();
  if (ended) return ended;
  if (peerStatus && controller.state.mode === "online" && peerStatus.status !== "connected") {
    if (peerStatus.status === "waiting") return tr("waiting");
    if (peerStatus.status === "reconnecting") return tr("reconnecting");
    if (peerStatus.status === "error") return peerStatus.error || tr("peerError");
    return tr("connecting");
  }
  if (controller.state.mode === 'cpu' && controller.engineKind === 'nnue' && controller.error) return tr('nnueError');
  if (controller.state.mode === 'cpu' && controller.paused) return tr('aiPaused');
  if (controller.state.position.inCheck(controller.state.position.turn)) return tr("check");
  if (controller.state.thinking) return tr("cpuThinking");
  if (controller.state.mode === "cpu") return controller.state.position.turn === controller.state.humanColor ? tr("yourTurn") : tr("cpuThinking");
  if (controller.state.mode === "online") return controller.state.position.turn === onlineColor ? tr("yourTurn") : tr("theirTurn");
  return controller.state.position.turn === BLACK ? tr("black") : tr("white");
}

function gameView(): string {
  const position = controller.state.position;
  const roomTools = controller.state.mode === "online" && peer ? `<div class="room-chip"><b>${peer.room}</b><button data-action="copy-invite">${tr("copyInvite")}</button></div>` : "";
  const progress = controller.state.progress ? `<span>${tr("depth")} ${controller.state.progress.depth} · ${controller.state.progress.nodes.toLocaleString()} ${tr("nodes")}</span>` : "";
  return `${topBar()}<main class="game-shell"><section class="play-column">
    <div class="game-status"><span class="turn-dot ${position.turn === BLACK ? "black" : "white"}"></span><strong>${statusText()}</strong>${progress}${roomTools}</div>
    ${handView(position, flipped ? BLACK : WHITE)}${boardView(position)}${handView(position, flipped ? WHITE : BLACK)}
    <div class="mobile-actions">${gameActions()}</div></section>
    <aside class="side-panel"><div class="panel-heading"><h2>${tr("moves")}</h2><span>${controller.state.history.length}</span></div>
      <ol class="move-list">${controller.state.history.length ? controller.state.history.map((move, index) => `<li class="${index === controller.state.history.length - 1 ? "current" : ""}"><span>${index + 1}</span><b>${move}</b></li>`).join("") : `<li class="empty">${tr("noMoves")}</li>`}</ol>
      <div class="desktop-actions">${gameActions()}</div></aside></main>${promotionModal()}`;
}

function gameActions(): string {
  const canUndo = controller.state.mode !== "online" && controller.state.history.length > 0;
  return `<div class="game-actions"><button data-action="undo" ${canUndo ? "" : "disabled"}>↶ <span>${tr("undo")}</span></button>
    <button data-action="flip">↻ <span>${tr("flip")}</span></button><button data-action="resign" class="danger">⚑ <span>${tr("resign")}</span></button>
    ${canInteract() && controller.state.position.canDeclareEnteringKing() && !controller.state.outcome ? `<button data-action="declare">入 <span>${tr("declare")}</span></button>` : ""}
    <button data-action="restart">＋ <span>${tr("restart")}</span></button></div>`;
}

function promotionModal(): string {
  if (!promotionChoices) return "";
  return `<div class="modal-backdrop"><div class="modal"><div class="promotion-piece">成</div><h2>${tr("promote")}</h2>
    <div class="modal-buttons"><button class="secondary" data-action="promotion-no">${tr("no")}</button><button class="primary" data-action="promotion-yes">${tr("yes")}</button></div></div></div>`;
}

function aboutView(): string {
  return `${topBar()}<main class="about-shell"><button class="back plain" data-action="home">← ${tr("home")}</button><article>
    <div class="about-mark">王</div><h1>AlphaSho Web</h1><p>Version 0.1.0</p>
    <h2>Software</h2><p>The Web application is open source under the MIT License. The desktop Python application remains a separate project.</p>
    <h2>Shogi pieces</h2><p>Piece artwork is from <a href="https://sunfish-shogi.github.io/shogi-images/" target="_blank" rel="noreferrer">Sunfish Shogi Images</a>, dedicated to the public domain under CC0 1.0.</p>
    <h2>Online play</h2><p>Online rooms use PeerJS and WebRTC. The public PeerServer helps peers discover each other; moves travel directly between the two browsers. Rooms are casual and are not stored.</p>
    <h2>AlphaSho NN 1</h2><p>${tr("nnueVersion")}</p><p>YaneuraOu V9.00 WebAssembly: GPLv3. <a href="${import.meta.env.BASE_URL}nnue/engine/COPYING.txt">License</a> · <a href="${import.meta.env.BASE_URL}nnue/engine/source.tar.gz">Corresponding patched source</a> · <a href="https://github.com/yaneurao/YaneuraOu/tree/V9.00">Upstream source</a>. Model weights trained by AlphaSho; no teacher weights imported.</p><p><a href="${import.meta.env.BASE_URL}benchmark.html">${tr("benchmark")}</a></p></article></main>`;
}

function render(): void {
  document.documentElement.lang = language === "zh" ? "zh-Hans" : language;
  app.innerHTML = screen === "home" ? homeView() : screen === "setup-cpu" ? cpuSetup() : screen === "setup-online" ? onlineSetup() : screen === "about" ? aboutView() : gameView();
  if (import.meta.env.MODE === 'nn-preview') {
    const notice = document.createElement('p'); notice.className = 'preview-notice';
    notice.textContent = language === 'zh' ? 'NN独立预览 · 手机验收与难度筛选进行中 · 正式网站未更新' : language === 'ja' ? 'NNプレビュー · 実機検証中 · 正式サイトは未更新' : 'NN preview · Device acceptance and difficulty screening pending · Production unchanged';
    app.querySelector('main')?.prepend(notice);
  }
  if (toast) app.insertAdjacentHTML("beforeend", `<div class="toast">${escapeHtml(toast)}${updateAvailable ? `<button data-action="refresh-update">${tr("refresh")}</button>` : ""}</div>`);
  if (screen === "game" && controller.engineKind === "nnue" && controller.state.mode === "cpu") {
    const panel = document.createElement("aside"); panel.className = "nnue-status";
    const label = document.createElement("p"); label.textContent = controller.error ? tr("nnueError") + ": " + controller.error : controller.loading !== null ? tr("loadingModel") + ` ${controller.loading}%` : tr("nnueLabel") + ` · ${tr("nnLevel")} ${controller.nnLevel}`;
    panel.append(label);
      const exportButton = document.createElement('button'); exportButton.dataset.action = 'export-nn-game'; exportButton.textContent = language === 'zh' ? '导出验收棋局' : language === 'ja' ? '対局記録を書き出す' : 'Export game record'; panel.append(exportButton);
    if (controller.error || controller.paused) {
      for (const [action, text] of [["retry-nnue", tr("retryAi")], ["use-abp", "ABP"]]) { const button = document.createElement("button"); button.dataset.action = action; button.textContent = text!; panel.append(button); }
    }
    app.querySelector("main")?.prepend(panel);
  }
  const list = app.querySelector(".move-list .current"); list?.scrollIntoView({ block: "nearest" });
}

function chooseMoveAt(square: number): void {
  if (!canInteract()) return;
  const position = controller.state.position;
  const candidates = selection.hand ? controller.legalMovesFrom(null, selection.hand).filter((move) => move.to === square) :
    selection.square !== null ? controller.legalMovesFrom(selection.square).filter((move) => move.to === square) : [];
  if (candidates.length) {
    if (candidates.length > 1) { promotionChoices = candidates; render(); return; }
    submitMove(candidates[0]!); return;
  }
  const piece = position.board[square] ?? 0;
  if (piece && colorOf(piece) === position.turn) selection = { square, hand: null };
  else selection = { square: null, hand: null };
  render();
}

function submitMove(move: Move): void {
  playSound(); selection = { square: null, hand: null }; promotionChoices = null;
  if (controller.state.mode === "online" && peer) {
    const sfen = controller.state.position.sfen(); const history = [...controller.state.history];
    peer.proposeMove(sfen, history, controller.state.position.moveToUsi(move));
  } else controller.play(move);
}

function startOnline(room: string, role: "host" | "guest", preferred: Color | "random" = "random"): void {
  peer?.close();
  controller.start("online");
  screen = "game"; peerStatus = { role, color: null, status: "connecting", room }; onlineColor = null;
  peer = new PeerSession(room, role, {
    onStatus: (status) => { peerStatus = status; if (status.color !== null) onlineColor = status.color; render(); },
    onState: (state) => {
      onlineColor = peer?.localColor ?? null;
      controller.load(state.sfen, state.history, "online", onlineColor, state.outcome);
      peer?.seedState(state.sfen, state.history); render();
    },
    onRematch: (accepted) => { if (accepted) { controller.start("online"); peer?.seedState(controller.state.position.sfen(), []); } }
  });
  peer.start(preferred); render();
  const url = new URL(location.href); url.searchParams.set("room", room); history.replaceState(null, "", url);
}

app.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>("[data-action], [data-square], [data-hand]");
  if (!target) return;
  if (target.dataset.square !== undefined) { chooseMoveAt(Number(target.dataset.square)); return; }
  if (target.dataset.hand !== undefined && canInteract()) {
    const hand = Number(target.dataset.hand) as PieceType;
    selection = selection.hand === hand ? { square: null, hand: null } : { square: null, hand }; render(); return;
  }
  const action = target.dataset.action;
  if (action === "home") { controller.pauseCpu(); peer?.close(); peer = null; peerStatus = null; screen = "home"; history.replaceState(null, "", location.pathname); render(); }
  else if (action === "retry-nnue") { controller.retryCpu(); }
  else if (action === 'export-nn-game') { const url = URL.createObjectURL(new Blob([JSON.stringify(controller.exportNnGame(), null, 2)], { type: 'application/json' })); const link = document.createElement('a'); link.href = url; link.download = 'alphasho-nn-game.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
  else if (action === "use-abp") { controller.useAbp(); render(); }
  else if (action === "setup-cpu") { screen = "setup-cpu"; render(); }
  else if (action === "setup-online") { screen = "setup-online"; render(); }
  else if (action === "start-local") { controller.start("local"); screen = "game"; render(); }
  else if (action === "start-cpu") {
    const side = app.querySelector<HTMLInputElement>('input[name="side"]:checked')?.value ?? "0";
    const difficulty = (app.querySelector<HTMLInputElement>('input[name="difficulty"]:checked')?.value ?? "normal") as Difficulty;
    const color = side === "random" ? (crypto.getRandomValues(new Uint8Array(1))[0]! & 1) as Color : Number(side) as Color;
    controller.engineKind = setupEngine; controller.nnLevel = Number(app.querySelector<HTMLSelectElement>('select[name="nn-level"]')?.value ?? 5);
    controller.start("cpu", color, difficulty); screen = "game"; flipped = color === WHITE; render();
  } else if (action === "create-room") {
    const side = app.querySelector<HTMLInputElement>('input[name="host-side"]:checked')?.value ?? "random";
    startOnline(createRoomCode(), "host", side === "random" ? "random" : Number(side) as Color);
  } else if (action === "join-room") {
    const code = app.querySelector<HTMLInputElement>("#room-code")?.value.trim().toUpperCase();
    if (code && /^[A-Z2-9]{10}$/.test(code)) startOnline(code, "guest");
  } else if (action === "copy-invite" && peer) {
    const url = new URL(location.href); url.searchParams.set("room", peer.room);
    void navigator.clipboard.writeText(url.toString()).then(() => setToast(tr("copied")));
  } else if (action === "undo") { selection = { square: null, hand: null }; controller.undo(); }
  else if (action === "flip") { flipped = !flipped; render(); }
  else if (action === "resign") {
    if (peer) peer.resign();
    else controller.resign(controller.state.mode === "cpu" ? controller.state.humanColor ?? controller.state.position.turn : controller.state.position.turn);
  }
  else if (action === "declare") {
    if (peer) peer.declare(controller.state.position.sfen(), controller.state.history);
    else controller.declareEnteringKing();
  }
  else if (action === "restart") { if (peer) peer.rematch(true); else controller.start(controller.state.mode, controller.state.humanColor, controller.difficulty); }
  else if (action === "sound") { sound = !sound; localStorage.setItem("alphasho.sound", sound ? "on" : "off"); render(); }
  else if (action === "refresh-update" && activateUpdate) { void activateUpdate(true); }
  else if (action === "about") { screen = "about"; render(); }
  else if (action === "promotion-yes" && promotionChoices) submitMove(promotionChoices.find((move) => move.promote) ?? promotionChoices[0]!);
  else if (action === "promotion-no" && promotionChoices) submitMove(promotionChoices.find((move) => !move.promote) ?? promotionChoices[0]!);
});

app.addEventListener("change", (event) => {
  const select = event.target as HTMLSelectElement;
  if (select.dataset.action === "language") { language = select.value as Language; localStorage.setItem("alphasho.language", language); render(); }
});

function playSound(): void {
  if (!sound) return;
  try {
    const AudioContextClass = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextClass) return;
    const context = new AudioContextClass(); const oscillator = context.createOscillator(); const gain = context.createGain();
    oscillator.type = "sine"; oscillator.frequency.value = 360; gain.gain.setValueAtTime(0.07, context.currentTime); gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.06);
    oscillator.connect(gain).connect(context.destination); oscillator.start(); oscillator.stop(context.currentTime + 0.06);
  } catch {
    // Audio support must never block a legal move (notably on iOS PWA/WebKit).
  }
}

function escapeHtml(value: string): string { const div = document.createElement("div"); div.textContent = value; return div.innerHTML; }

app.addEventListener("change", event => { const el = event.target as HTMLInputElement; if (el.name === "engine") { setupEngine = el.value as EngineKind; render(); } });
document.addEventListener("visibilitychange", () => { if (controller.state.mode !== "cpu") return; if (document.hidden) controller.pauseCpu(); else if (controller.engineKind === "abp") controller.retryCpu(); });
if ("serviceWorker" in navigator) {
  const isolationKey = 'alphasho.isolation-reloaded-v1:' + import.meta.env.BASE_URL;
  const isolate = () => { if (!crossOriginIsolated && !sessionStorage.getItem(isolationKey)) { sessionStorage.setItem(isolationKey, "1"); location.reload(); } };
  navigator.serviceWorker.addEventListener("controllerchange", isolate);
  void navigator.serviceWorker.ready.then(() => { if (navigator.serviceWorker.controller) isolate(); });
}
activateUpdate = registerSW({
  onOfflineReady: () => setToast(tr("offlineReady")),
  onNeedRefresh: () => {
    updateAvailable = true;
    toast = tr("updateReady");
    render();
  }
});

const invitation = new URLSearchParams(location.search).get("room");
if (invitation) screen = "setup-online";
else {
  try {
    const saved = JSON.parse(localStorage.getItem("alphasho.game") ?? "null") as {
      schemaVersion?: number; mode?: Mode; sfen?: string; history?: string[]; humanColor?: Color | null;
      outcome?: import("./engine/types").GameOutcome | null; difficulty?: Difficulty; engineKind?: EngineKind; nnLevel?: number; rookTarget?: RookRoute;
    } | null;
    if (saved?.schemaVersion === 1 && saved.mode && saved.sfen && Array.isArray(saved.history)) {
      controller.difficulty = saved.difficulty ?? "normal"; controller.engineKind = saved.engineKind === "nnue" ? "nnue" : "abp"; controller.nnLevel = Math.max(1, Math.min(10, saved.nnLevel ?? 5)); controller.rookTarget = saved.rookTarget;
      screen = "game";
      controller.load(saved.sfen, saved.history, saved.mode, saved.humanColor ?? null, saved.outcome ?? null);
    }
  } catch { localStorage.removeItem("alphasho.game"); }
}
render();
