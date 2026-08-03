import { Color, GameOutcome } from "../engine/types";
import { Position } from "../engine/position";

export const PROTOCOL_VERSION = 1 as const;
export const APP_VERSION = "0.1.0";

interface Envelope { protocolVersion: 1; appVersion: string; room: string }
export interface HelloMessage extends Envelope { type: "hello"; role: "host" | "guest" }
export interface StartMessage extends Envelope {
  type: "start"; guestId: string; hostColor: Color; sequence: 0; sfen: string; history: string[]; stateHash: string;
}
export interface MoveProposalMessage extends Envelope {
  type: "move-proposal"; sequence: number; usi: string; beforeHash: string;
}
export interface StateMessage extends Envelope {
  type: "state"; sequence: number; sfen: string; history: string[]; lastMove: string | null;
  stateHash: string; outcome: GameOutcome | null;
}
export interface ResignMessage extends Envelope { type: "resign"; sequence: number }
export interface DeclareMessage extends Envelope { type: "declare"; sequence: number; beforeHash: string }
export interface RematchMessage extends Envelope { type: "rematch"; accepted: boolean }
export interface PingMessage extends Envelope { type: "ping" | "pong"; timestamp: number }
export interface ErrorMessage extends Envelope { type: "error"; code: string; message: string; recoverable: boolean }

export type NetworkMessage = HelloMessage | StartMessage | MoveProposalMessage | StateMessage |
  ResignMessage | DeclareMessage | RematchMessage | PingMessage | ErrorMessage;

export function envelope(room: string): Envelope {
  return { protocolVersion: PROTOCOL_VERSION, appVersion: APP_VERSION, room };
}

export function isNetworkMessage(value: unknown): value is NetworkMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Partial<NetworkMessage>;
  const types = new Set(["hello", "start", "move-proposal", "state", "resign", "declare", "rematch", "ping", "pong", "error"]);
  return message.protocolVersion === PROTOCOL_VERSION && message.appVersion === APP_VERSION &&
    typeof message.room === "string" && typeof message.type === "string" && types.has(message.type);
}

export type ProposalValidation = { ok: true } | { ok: false; code: "sequence" | "hash" | "turn" | "illegal-move" };
export function validateMoveProposal(position: Position, hostColor: Color, sequence: number,
  currentSequence: number, beforeHash: string, usi: string): ProposalValidation {
  if (sequence !== currentSequence + 1) return { ok: false, code: "sequence" };
  if (beforeHash !== position.hashKey()) return { ok: false, code: "hash" };
  if (position.turn === hostColor) return { ok: false, code: "turn" };
  if (!position.legalUsiMoves().includes(usi)) return { ok: false, code: "illegal-move" };
  return { ok: true };
}

export function createRoomCode(): string {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  const bytes = new Uint8Array(10);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (value) => alphabet[value % alphabet.length]).join("");
}
