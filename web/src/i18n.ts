export type Language = "ja" | "zh" | "en";

const messages = {
  en: {
    subtitle: "A thoughtful game, entirely in your browser", vsCpu: "Play the computer", local: "Two players here",
    online: "Online room", easy: "Easy", normal: "Normal", hard: "Hard", black: "Black (first)",
    white: "White (second)", random: "Random", start: "Start game", createRoom: "Create a room",
    joinRoom: "Join room", roomCode: "Room code", copyInvite: "Copy invite", copied: "Invite copied",
    waiting: "Waiting for a friend…", connecting: "Connecting…", reconnecting: "Connection lost — reconnecting…",
    yourTurn: "Your turn", theirTurn: "Opponent's turn", cpuThinking: "Computer is thinking", check: "Check",
    resign: "Resign", restart: "New game", undo: "Undo", flip: "Flip board", sound: "Sound", declare: "Declare win",
    home: "Home", promote: "Promote this piece?", yes: "Promote", no: "Do not promote",
    winnerBlack: "Black wins", winnerWhite: "White wins", draw: "Draw", checkmate: "Checkmate",
    repetition: "Fourfold repetition", perpetual: "Perpetual check", enteringKing: "Entering-king declaration",
    resigned: "Resignation", depth: "Depth", nodes: "nodes", roomEnded: "The room has ended",
    peerError: "Online rooms are unavailable right now. Local games still work.", about: "About & licenses",
    offlineReady: "AlphaSho is ready for offline play.", updateReady: "A new version is ready.", refresh: "Refresh",
    chooseSide: "Choose your side", chooseDifficulty: "Choose difficulty", hostSide: "Host plays",
    moves: "Moves", hand: "Pieces in hand", noMoves: "No moves yet", onlineNote: "Casual peer-to-peer game · no account required"
  },
  zh: {
    subtitle: "一盘好棋，完全运行在浏览器里", vsCpu: "人机对战", local: "本地双人", online: "在线房间",
    easy: "简单", normal: "普通", hard: "困难", black: "先手", white: "后手", random: "随机",
    start: "开始对局", createRoom: "创建房间", joinRoom: "加入房间", roomCode: "房间码",
    copyInvite: "复制邀请链接", copied: "邀请已复制", waiting: "正在等待朋友加入……", connecting: "正在连接……",
    reconnecting: "连接中断，正在重连……", yourTurn: "轮到你", theirTurn: "轮到对方", cpuThinking: "电脑思考中",
    check: "王手", resign: "投降", restart: "重新开始", undo: "悔棋", flip: "翻转棋盘", sound: "音效", declare: "入玉宣胜",
    home: "返回首页", promote: "要升变吗？", yes: "升变", no: "不升变", winnerBlack: "先手胜",
    winnerWhite: "后手胜", draw: "和棋", checkmate: "将死", repetition: "千日手", perpetual: "连续王手千日手",
    enteringKing: "入玉宣言", resigned: "投降", depth: "深度", nodes: "节点", roomEnded: "房间已经结束",
    peerError: "在线房间暂时不可用，本地和人机对战不受影响。", about: "关于与许可",
    offlineReady: "AlphaSho 已可离线使用。", updateReady: "新版本已准备好。", refresh: "刷新",
    chooseSide: "选择执子", chooseDifficulty: "选择难度", hostSide: "房主执子", moves: "棋谱",
    hand: "持驹", noMoves: "尚未行棋", onlineNote: "休闲点对点对局 · 无需账号"
  },
  ja: {
    subtitle: "ブラウザだけで楽しむ、静かな一局", vsCpu: "コンピューター対局", local: "この端末で二人",
    online: "オンラインルーム", easy: "やさしい", normal: "ふつう", hard: "むずかしい", black: "先手",
    white: "後手", random: "ランダム", start: "対局開始", createRoom: "ルームを作る", joinRoom: "参加する",
    roomCode: "ルームコード", copyInvite: "招待リンクをコピー", copied: "コピーしました", waiting: "相手を待っています…",
    connecting: "接続しています…", reconnecting: "接続が切れました。再接続中…", yourTurn: "あなたの手番",
    theirTurn: "相手の手番", cpuThinking: "思考中", check: "王手", resign: "投了", restart: "新しい対局",
    undo: "待った", flip: "盤を反転", sound: "効果音", declare: "入玉宣言", home: "ホーム", promote: "成りますか？",
    yes: "成る", no: "成らない", winnerBlack: "先手の勝ち", winnerWhite: "後手の勝ち", draw: "引き分け",
    checkmate: "詰み", repetition: "千日手", perpetual: "連続王手の千日手", enteringKing: "入玉宣言",
    resigned: "投了", depth: "深さ", nodes: "局面", roomEnded: "ルームは終了しました",
    peerError: "オンラインルームを利用できません。ローカル対局は引き続き遊べます。", about: "このアプリとライセンス",
    offlineReady: "オフラインでも遊べます。", updateReady: "新しいバージョンがあります。", refresh: "更新",
    chooseSide: "手番を選ぶ", chooseDifficulty: "強さを選ぶ", hostSide: "ホストの手番", moves: "棋譜",
    hand: "持ち駒", noMoves: "まだ指し手はありません", onlineNote: "気軽なP2P対局 · アカウント不要"
  }
} as const;

export type MessageKey = keyof typeof messages.en;
export function t(language: Language, key: MessageKey): string { return messages[language][key] ?? messages.en[key]; }

export function detectLanguage(): Language {
  const saved = localStorage.getItem("alphasho.language");
  if (saved === "ja" || saved === "zh" || saved === "en") return saved;
  const locale = navigator.language.toLowerCase();
  return locale.startsWith("ja") ? "ja" : locale.startsWith("zh") ? "zh" : "en";
}
