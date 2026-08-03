import Peer, { DataConnection } from "peerjs";
import { Position } from "../engine/position";
import { BLACK, Color, GameOutcome, opposite } from "../engine/types";
import {
  APP_VERSION, NetworkMessage, PROTOCOL_VERSION, StateMessage, envelope, isNetworkMessage, validateMoveProposal
} from "./protocol";

export type SessionStatus = "connecting" | "waiting" | "connected" | "reconnecting" | "closed" | "error";
export interface SessionSnapshot {
  role: "host" | "guest";
  color: Color | null;
  status: SessionStatus;
  room: string;
  error?: string;
}

interface Callbacks {
  onStatus: (snapshot: SessionSnapshot) => void;
  onState: (message: StateMessage) => void;
  onRematch: (accepted: boolean) => void;
}

const RECONNECT_MS = 30_000;

export class PeerSession {
  private peer: Peer | null = null;
  private connection: DataConnection | null = null;
  private sequence = 0;
  private reconnectTimer: number | null = null;
  private heartbeat: number | null = null;
  private guestId = "";
  private hostColor: Color = BLACK;

  constructor(readonly room: string, readonly role: "host" | "guest", private readonly callbacks: Callbacks) {}

  start(preferredHostColor: Color | "random" = "random"): void {
    this.hostColor = preferredHostColor === "random" ? ((crypto.getRandomValues(new Uint8Array(1))[0] ?? 0) & 1) as Color : preferredHostColor;
    this.emit("connecting");
    this.peer = this.role === "host" ? new Peer(`alphasho-${this.room}`, { debug: 0 }) : new Peer({ debug: 0 });
    this.peer.on("open", () => {
      if (this.role === "host") this.emit("waiting");
      else this.bind(this.peer!.connect(`alphasho-${this.room}`, { reliable: true, serialization: "json" }));
    });
    this.peer.on("connection", (connection) => {
      if (this.role !== "host") { connection.close(); return; }
      if (this.connection?.open) {
        connection.on("open", () => connection.send({ ...envelope(this.room), type: "error", code: "room-full", message: "Room already has two players", recoverable: false }));
        return;
      }
      this.bind(connection);
    });
    this.peer.on("error", (error) => this.emit("error", error.message));
    this.peer.on("disconnected", () => this.beginReconnect());
  }

  private bind(connection: DataConnection): void {
    this.connection = connection;
    connection.on("open", () => {
      this.stopReconnect();
      this.emit("connected");
      connection.send({ ...envelope(this.room), type: "hello", role: this.role });
      if (this.role === "host") {
        this.guestId = connection.peer;
        if (this.lastState) this.send(this.lastState);
        else {
          const position = new Position();
          this.seedState(position.sfen(), []);
          connection.send({ ...envelope(this.room), type: "start", guestId: this.guestId,
            hostColor: this.hostColor, sequence: 0, sfen: position.sfen(), history: [], stateHash: position.hashKey() });
        }
      }
      this.heartbeat = window.setInterval(() => this.send({ ...envelope(this.room), type: "ping", timestamp: Date.now() }), 10_000);
    });
    connection.on("data", (data) => this.receive(data));
    connection.on("close", () => this.beginReconnect());
    connection.on("error", (error) => this.emit("error", error.message));
  }

  private receive(raw: unknown): void {
    if (!isNetworkMessage(raw)) {
      this.send({ ...envelope(this.room), type: "error", code: "protocol-version",
        message: `Requires protocol ${PROTOCOL_VERSION} / app ${APP_VERSION}`, recoverable: false });
      return;
    }
    const message = raw as NetworkMessage;
    if (message.room !== this.room) return;
    if (message.type === "ping") this.send({ ...envelope(this.room), type: "pong", timestamp: message.timestamp });
    else if (message.type === "start" && this.role === "guest") {
      this.hostColor = message.hostColor;
      this.sequence = 0;
      this.callbacks.onState({ ...envelope(this.room), type: "state", sequence: 0, sfen: message.sfen,
        history: message.history, lastMove: null, stateHash: message.stateHash, outcome: null });
      this.emit("connected");
    } else if (message.type === "move-proposal" && this.role === "host") this.validateProposal(message.sequence, message.usi, message.beforeHash);
    else if (message.type === "state") {
      if (message.sequence < this.sequence) return;
      const position = this.rebuild(message.sfen, message.history);
      if (position.hashKey() !== message.stateHash) { this.requestSync("state-hash"); return; }
      this.sequence = message.sequence;
      this.callbacks.onState(message);
    } else if (message.type === "resign" && this.role === "host" && message.sequence === this.sequence) {
      const winner = this.hostColor;
      this.broadcastCurrent({ winner, reason: "resignation" });
    } else if (message.type === "declare" && this.role === "host" && message.sequence === this.sequence) {
      const last = this.lastState;
      if (!last) return;
      const position = this.rebuild(last.sfen, last.history);
      if (message.beforeHash === position.hashKey() && position.turn !== this.hostColor && position.canDeclareEnteringKing())
        this.broadcastCurrent({ winner: position.turn, reason: "entering-king" });
    } else if (message.type === "rematch" && message.accepted) {
      if (this.role === "host") this.resetMatch(true);
      else this.callbacks.onRematch(true);
    }
    else if (message.type === "error") this.emit(message.recoverable ? "connected" : "error", message.message);
  }

  proposeMove(sfen: string, history: string[], usi: string): void {
    const position = this.rebuild(sfen, history);
    if (this.role === "host") this.applyHostMove(position, history, usi);
    else this.send({ ...envelope(this.room), type: "move-proposal", sequence: this.sequence + 1, usi, beforeHash: position.hashKey() });
  }

  private validateProposal(sequence: number, usi: string, beforeHash: string): void {
    const last = this.lastState;
    if (!last) { this.sendState(); return; }
    const position = this.rebuild(last.sfen, last.history);
    const validation = validateMoveProposal(position, this.hostColor, sequence, this.sequence, beforeHash, usi);
    if (!validation.ok) {
      this.send({ ...envelope(this.room), type: "error", code: validation.code, message: "Move rejected by host", recoverable: true });
      this.sendState();
      return;
    }
    this.applyAndBroadcast(position, last.history, usi);
  }

  private applyHostMove(position: Position, history: string[], usi: string): void {
    if (position.turn !== this.hostColor || !position.legalUsiMoves().includes(usi)) throw new Error("Not a legal host move");
    this.applyAndBroadcast(position, history, usi);
  }

  private lastState: StateMessage | null = null;
  private applyAndBroadcast(position: Position, history: string[], usi: string): void {
    position.pushUsi(usi);
    this.sequence += 1;
    const state: StateMessage = { ...envelope(this.room), type: "state", sequence: this.sequence,
      sfen: position.sfen(), history: [...history, usi], lastMove: usi, stateHash: position.hashKey(), outcome: position.outcome() };
    this.lastState = state;
    this.callbacks.onState(state);
    this.send(state);
  }

  private broadcastCurrent(outcome: GameOutcome): void {
    if (!this.lastState) return;
    const state = { ...this.lastState, outcome };
    this.lastState = state;
    this.callbacks.onState(state);
    this.send(state);
  }

  seedState(sfen: string, history: string[]): void {
    const position = this.rebuild(sfen, history);
    this.lastState = { ...envelope(this.room), type: "state", sequence: this.sequence,
      sfen, history, lastMove: history.at(-1) ?? null, stateHash: position.hashKey(), outcome: position.outcome() };
  }

  get localColor(): Color { return this.role === "host" ? this.hostColor : opposite(this.hostColor); }

  resign(): void {
    if (this.role === "host") this.broadcastCurrent({ winner: opposite(this.hostColor), reason: "resignation" });
    else this.send({ ...envelope(this.room), type: "resign", sequence: this.sequence });
  }

  declare(sfen: string, history: string[]): void {
    const position = this.rebuild(sfen, history);
    if (!position.canDeclareEnteringKing()) return;
    if (this.role === "host" && position.turn === this.hostColor) this.broadcastCurrent({ winner: position.turn, reason: "entering-king" });
    else this.send({ ...envelope(this.room), type: "declare", sequence: this.sequence, beforeHash: position.hashKey() });
  }

  rematch(accepted: boolean): void {
    if (accepted && this.role === "host") this.resetMatch(true);
    else this.send({ ...envelope(this.room), type: "rematch", accepted });
  }

  private resetMatch(notifyPeer: boolean): void {
    this.sequence = 0;
    const position = new Position();
    this.seedState(position.sfen(), []);
    this.callbacks.onState(this.lastState!);
    if (notifyPeer) {
      this.send({ ...envelope(this.room), type: "rematch", accepted: true });
      this.send(this.lastState!);
    }
  }

  private requestSync(code: string): void {
    this.send({ ...envelope(this.room), type: "error", code, message: "State mismatch; requesting resync", recoverable: true });
    if (this.role === "host") this.sendState();
  }
  private sendState(): void { if (this.lastState) this.send(this.lastState); }
  private send(message: NetworkMessage): void { if (this.connection?.open) this.connection.send(message); }
  private rebuild(sfen: string, history: string[]): Position {
    return history.length ? Position.fromHistory(history, sfen) : new Position(sfen);
  }

  private beginReconnect(): void {
    if (this.reconnectTimer !== null || this.closed) return;
    this.emit("reconnecting");
    const started = Date.now();
    this.reconnectTimer = window.setInterval(() => {
      if (Date.now() - started > RECONNECT_MS) { this.stopReconnect(); this.emit("closed", "Reconnect window expired"); return; }
      if (this.peer?.disconnected) this.peer.reconnect();
      if (this.role === "guest" && this.peer?.open && !this.connection?.open) this.bind(this.peer.connect(`alphasho-${this.room}`, { reliable: true, serialization: "json" }));
    }, 1_000);
  }
  private stopReconnect(): void { if (this.reconnectTimer !== null) window.clearInterval(this.reconnectTimer); this.reconnectTimer = null; }
  private closed = false;
  close(): void {
    this.closed = true;
    this.stopReconnect();
    if (this.heartbeat !== null) window.clearInterval(this.heartbeat);
    this.connection?.close(); this.peer?.destroy(); this.emit("closed");
  }
  private emit(status: SessionStatus, error?: string): void {
    this.callbacks.onStatus({ role: this.role, color: status === "connected" ? this.localColor : null, status, room: this.room, error });
  }
}
