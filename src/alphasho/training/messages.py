"""Language-neutral training event messages rendered for human-readable logs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from string import Formatter
from typing import Any


class TrainingLanguage(StrEnum):
    ENGLISH = "en"
    CHINESE = "zh"
    JAPANESE = "ja"

    @classmethod
    def parse(cls, value: str | TrainingLanguage) -> TrainingLanguage:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().replace("-", "_").lower()
        aliases = {
            "en": cls.ENGLISH,
            "en_us": cls.ENGLISH,
            "english": cls.ENGLISH,
            "zh": cls.CHINESE,
            "zh_cn": cls.CHINESE,
            "chinese": cls.CHINESE,
            "中文": cls.CHINESE,
            "ja": cls.JAPANESE,
            "ja_jp": cls.JAPANESE,
            "japanese": cls.JAPANESE,
            "日本語": cls.JAPANESE,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(
                "output_language: choose en, zh, or ja; "
                f"got {value!r}"
            ) from exc


def _translations(en: str, zh: str, ja: str) -> dict[TrainingLanguage, str]:
    return {
        TrainingLanguage.ENGLISH: en,
        TrainingLanguage.CHINESE: zh,
        TrainingLanguage.JAPANESE: ja,
    }


MESSAGE_CATALOG: dict[str, dict[TrainingLanguage, str]] = {
    "error.trainer-lock-conflict": _translations(
        "Training cannot start: {detail}",
        "无法启动训练：{detail}",
        "トレーニングを開始できません：{detail}",
    ),
    "error.incompatible-run": _translations(
        "Training cannot resume because persisted artifacts are incompatible: {detail}",
        "无法恢复训练，因为现有训练产物不兼容：{detail}",
        "保存済み artifact に互換性がないため再開できません：{detail}",
    ),
    "completed.already-complete": _translations(
        "Training is already complete: {completed_iterations}/{iterations} iterations",
        "训练已经完成：{completed_iterations}/{iterations} iterations",
        "トレーニングは完了済みです：{completed_iterations}/{iterations} iterations",
    ),
    "stopped.stopped": _translations(
        "Training stopped safely. Completed games, steps, and evaluation results "
        "were saved; the unfinished item will be repeated after resume.",
        "训练已安全停止；已完成的对局、step 和评估结果已经保存，未完成项目将在恢复后重做。",
        "トレーニングを安全に停止しました。完了した対局、step、評価結果は保存され、"
        "未完了の項目は再開後にやり直されます。",
    ),
    "error.fatal-error": _translations(
        "Training stopped because of {error_type}: {detail}",
        "训练因错误停止：{error_type}: {detail}",
        "エラーによりトレーニングを停止しました：{error_type}: {detail}",
    ),
    "checkpoint.temporary-state-recovered": _translations(
        "Recovered a temporary state newer than the committed checkpoint: "
        "iteration={active_iteration}, selfplay={selfplay_completed}",
        "已恢复比正式 checkpoint 更新的临时训练状态："
        "iteration={active_iteration}，selfplay={selfplay_completed}",
        "正式な checkpoint より新しい一時状態を復元しました："
        "iteration={active_iteration}、selfplay={selfplay_completed}",
    ),
    "startup.resume": _translations(
        "Resumed checkpoint: completed iteration {completed_iterations}, "
        "current stage {current_stage}",
        "已恢复 checkpoint：完成 iteration {completed_iterations}，"
        "当前阶段 {current_stage}",
        "checkpoint を再開しました：完了 iteration {completed_iterations}、"
        "現在の stage {current_stage}",
    ),
    "startup.new-run": _translations(
        "No training state found; starting from the current best model.",
        "未发现训练状态，将从当前最佳模型开始。",
        "トレーニング状態が見つかりません。現在の best model から開始します。",
    ),
    "selfplay.iteration-start": _translations(
        "Starting iteration {iteration}/{iterations}",
        "开始 iteration {iteration}/{iterations}",
        "iteration {iteration}/{iterations} を開始",
    ),
    "evaluation.evaluation-skipped": _translations(
        "Evaluation skipped this iteration; candidates are checked every "
        "{check_frequency} iterations.",
        "本轮不评估；每 {check_frequency} 个 iteration 检查一次候选模型。",
        "この iteration の評価は省略します。候補モデルは "
        "{check_frequency} iterations ごとに確認します。",
    ),
    "selfplay.game-start": _translations(
        "Self-play game {game}/{games} started",
        "自我对弈 {game}/{games} 开始",
        "自己対局 {game}/{games} を開始",
    ),
    "selfplay.parallel-start": _translations(
        "Self-play scheduler started: {workers} workers, "
        "{pending_games} pending games, GPU batch up to {gpu_batch_size}",
        "自我对弈并行调度已启动：{workers} 个 worker，"
        "待完成 {pending_games} 局，GPU batch 上限 {gpu_batch_size}",
        "自己対局の並列実行を開始：{workers} workers、"
        "残り {pending_games} games、GPU batch 上限 {gpu_batch_size}",
    ),
    "selfplay.search-heartbeat": _translations(
        "Self-play {game}/{games}, ply {ply}: searching "
        "{completed_playouts}/{total_playouts} playouts",
        "自我对弈 {game}/{games}，第 {ply} 手搜索中："
        "{completed_playouts}/{total_playouts} playouts",
        "自己対局 {game}/{games}、{ply} 手目を探索中："
        "{completed_playouts}/{total_playouts} playouts",
    ),
    "selfplay.move-complete": _translations(
        "Ply {ply}: {usi}, {playouts_per_second:.1f} playouts/s",
        "第 {ply} 手 {usi}，{playouts_per_second:.1f} playouts/s",
        "{ply} 手目 {usi}、{playouts_per_second:.1f} playouts/s",
    ),
    "selfplay.game-complete": _translations(
        "Self-play {game}/{games} complete: {moves} plies, {winner_text}, "
        "{seconds:.1f}s, ETA {eta}",
        "自我对弈 {game}/{games} 完成：{moves} 手，结果 {winner_text}，"
        "本局 {seconds:.1f}s，ETA {eta}",
        "自己対局 {game}/{games} 完了：{moves} 手、結果 {winner_text}、"
        "{seconds:.1f}s、ETA {eta}",
    ),
    "selfplay.all-draw-warning": _translations(
        "Every self-play game in this iteration was a draw; check max moves, "
        "search volume, and model signal.",
        "本轮自我对弈全部为和棋；请检查最大手数、搜索量和模型信号。",
        "この iteration の自己対局はすべて引き分けでした。"
        "最大手数、探索量、モデル信号を確認してください。",
    ),
    "selfplay.selfplay-diagnostics": _translations(
        "Short-game diagnostics: terminals={terminal_counts_text}, "
        "mean captures={mean_captures:.2f}, "
        "mean |material diff|={mean_absolute_material_diff:.2f}, "
        "non-zero value={nonzero_value_rate:.1%}",
        "短局诊断：终局={terminal_counts_text}，平均吃子={mean_captures:.2f}，"
        "平均|材料差|={mean_absolute_material_diff:.2f}，"
        "非零 value={nonzero_value_rate:.1%}",
        "短局診断：終局={terminal_counts_text}、平均捕獲数={mean_captures:.2f}、"
        "平均|駒得差|={mean_absolute_material_diff:.2f}、"
        "非ゼロ value={nonzero_value_rate:.1%}",
    ),
    "training.small-replay-warning": _translations(
        "Replay contains {replay_samples} samples, below batch size "
        "{batch_size}; sampling with replacement.",
        "Replay 只有 {replay_samples} 个样本，小于 batch size "
        "{batch_size}；将有放回采样。",
        "Replay は {replay_samples} samples で batch size {batch_size} "
        "未満のため、復元抽出します。",
    ),
    "training.training-start": _translations(
        "Network training started: step {completed_steps}/{train_steps}, "
        "replay {replay_samples}/{replay_capacity}",
        "网络训练开始：step {completed_steps}/{train_steps}，"
        "replay {replay_samples}/{replay_capacity}",
        "ネットワーク学習を開始：step {completed_steps}/{train_steps}、"
        "replay {replay_samples}/{replay_capacity}",
    ),
    "training.train-step": _translations(
        "step {step}/{total_steps} loss={loss:.4f} "
        "(policy={policy_loss:.4f}, value={value_loss:.4f}) "
        "entropy={predicted_entropy:.3f} KL={kl:.5f} "
        "grad={gradient_norm:.3f} {samples_per_second:.1f} samples/s ETA {eta}",
        "step {step}/{total_steps} loss={loss:.4f} "
        "(policy={policy_loss:.4f}, value={value_loss:.4f}) "
        "entropy={predicted_entropy:.3f} KL={kl:.5f} "
        "grad={gradient_norm:.3f} {samples_per_second:.1f} samples/s ETA {eta}",
        "step {step}/{total_steps} loss={loss:.4f} "
        "(policy={policy_loss:.4f}, value={value_loss:.4f}) "
        "entropy={predicted_entropy:.3f} KL={kl:.5f} "
        "grad={gradient_norm:.3f} {samples_per_second:.1f} samples/s ETA {eta}",
    ),
    "training.entropy-warning": _translations(
        "Policy entropy is outside the monitoring range: {entropy:.4f}",
        "策略 entropy 超出监控范围：{entropy:.4f}",
        "Policy entropy が監視範囲外です：{entropy:.4f}",
    ),
    "training.training-complete": _translations(
        "Network training complete: loss={loss:.4f}, entropy={entropy:.3f}, "
        "{samples_per_second:.1f} samples/s",
        "网络训练完成：loss={loss:.4f}，entropy={entropy:.3f}，"
        "{samples_per_second:.1f} samples/s",
        "ネットワーク学習完了：loss={loss:.4f}、entropy={entropy:.3f}、"
        "{samples_per_second:.1f} samples/s",
    ),
    "evaluation.evaluation-start": _translations(
        "Candidate evaluation started: {completed_games}/{evaluation_games}, "
        "promotion threshold {promotion_score:.1%}",
        "候选评估开始：{completed_games}/{evaluation_games}，"
        "晋级阈值 {promotion_score:.1%}",
        "候補評価を開始：{completed_games}/{evaluation_games}、"
        "昇格しきい値 {promotion_score:.1%}",
    ),
    "evaluation.evaluation-progress-trimmed": _translations(
        "Evaluation target changed to {evaluation_games} games; retained "
        "{retained_games} of {previous_games} completed results.",
        "评估目标已改为 {evaluation_games} 局；已完成的 {previous_games} "
        "局中保留了 {retained_games} 局结果。",
        "評価数を {evaluation_games} 局に変更しました。完了済み "
        "{previous_games} 局のうち {retained_games} 局を保持しました。",
    ),
    "evaluation.parallel-start": _translations(
        "Evaluation scheduler started: {workers} workers, "
        "{pending_games} pending games, GPU batch up to {gpu_batch_size}",
        "评估并行调度已启动：{workers} 个 worker，"
        "待完成 {pending_games} 局，GPU batch 上限 {gpu_batch_size}",
        "評価の並列実行を開始：{workers} workers、"
        "残り {pending_games} games、GPU batch 上限 {gpu_batch_size}",
    ),
    "evaluation.search-heartbeat": _translations(
        "Evaluation game {game}, ply {ply}: "
        "{completed_playouts}/{total_playouts} playouts",
        "评估第 {game} 局、第 {ply} 手："
        "{completed_playouts}/{total_playouts} playouts",
        "評価第 {game} 局、{ply} 手目："
        "{completed_playouts}/{total_playouts} playouts",
    ),
    "evaluation.move-heartbeat": _translations(
        "Evaluation game {game}, ply {ply}: "
        "{playouts_per_second:.1f} playouts/s",
        "评估第 {game} 局、第 {ply} 手，"
        "{playouts_per_second:.1f} playouts/s",
        "評価第 {game} 局、{ply} 手目、"
        "{playouts_per_second:.1f} playouts/s",
    ),
    "evaluation.evaluation-game": _translations(
        "Evaluation {game}/{games}: candidate {candidate_wins} wins, "
        "best {best_wins} wins, {draws} draws, score {score:.3f}, ETA {eta}",
        "评估 {game}/{games}：候选 {candidate_wins} 胜，"
        "最佳 {best_wins} 胜，和棋 {draws}，得分 {score:.3f}，ETA {eta}",
        "評価 {game}/{games}：候補 {candidate_wins} 勝、"
        "best {best_wins} 勝、引き分け {draws}、score {score:.3f}、ETA {eta}",
    ),
    "evaluation.evaluation-complete": _translations(
        "Evaluation complete: candidate score {score:.3f}, threshold "
        "{promotion_score:.3f}; {promotion_text}",
        "评估完成：候选得分 {score:.3f}，阈值 {promotion_score:.3f}；"
        "{promotion_text}",
        "評価完了：候補 score {score:.3f}、しきい値 "
        "{promotion_score:.3f}；{promotion_text}",
    ),
    "evaluation.promotion-stall-warning": _translations(
        "The candidate has failed promotion {consecutive_no_promotion} times "
        "in a row; check the training signal and hyperparameters.",
        "候选模型已连续 {consecutive_no_promotion} 次未晋级，"
        "请检查训练信号和超参数。",
        "候補モデルは {consecutive_no_promotion} 回連続で昇格していません。"
        "学習信号とハイパーパラメータを確認してください。",
    ),
    "completed.iteration-complete": _translations(
        "iteration {iteration}/{iterations} complete in {elapsed}; "
        "loss={loss_text}; {iteration_outcome}",
        "iteration {iteration}/{iterations} 完成，耗时 {elapsed}；"
        "loss={loss_text}；{iteration_outcome}",
        "iteration {iteration}/{iterations} 完了、所要時間 {elapsed}；"
        "loss={loss_text}；{iteration_outcome}",
    ),
    "supervised.ingest-start": _translations(
        "Starting supervised game-record import.",
        "开始导入监督棋谱。",
        "教師棋譜のインポートを開始します。",
    ),
    "supervised.ingest-complete": _translations(
        "Supervised import complete: {games}/{files} games, {samples} samples, "
        "{skipped} skipped",
        "监督棋谱导入完成：{games}/{files} 局，{samples} 个样本，"
        "跳过 {skipped}",
        "教師棋譜のインポート完了：{games}/{files} 局、{samples} samples、"
        "{skipped} 件をスキップ",
    ),
    "supervised.warmup-complete": _translations(
        "Supervised warm-up complete: loss={loss:.4f}",
        "监督预热完成：loss={loss:.4f}",
        "教師あり warm-up 完了：loss={loss:.4f}",
    ),
    "startup.cpu-baseline-warning": _translations(
        "Baseline is running on CPU and will be extremely slow; CUDA Torch "
        "is recommended for production training.",
        "Baseline 正在使用 CPU，正式训练会非常慢；建议安装 CUDA Torch。",
        "Baseline を CPU で実行しています。正式学習には非常に時間がかかるため、"
        "CUDA Torch を推奨します。",
    ),
    "startup.model-ready": _translations(
        "Model ready: {parameters:,} parameters, device={device}",
        "模型已就绪：{parameters:,} 参数，device={device}",
        "モデル準備完了：{parameters:,} parameters、device={device}",
    ),
    "startup.startup": _translations(
        "AlphaSho Trainer started: profile={profile}, advanced config={config_path}",
        "AlphaSho Trainer 启动：profile={profile}，高级配置={config_path}",
        "AlphaSho Trainer 起動：profile={profile}、詳細設定={config_path}",
    ),
    "startup.settings": _translations(
        "Search: playouts={playouts}, c_puct={c_puct}, "
        "inference_batch={inference_batch_size}; data: "
        "games/iter={games_per_iteration}, buffer={replay_buffer_size:,}; "
        "training: batch={training_batch_size}, "
        "steps/iter={train_steps_per_iteration}, lr={learning_rate}",
        "搜索：playouts={playouts}，c_puct={c_puct}，"
        "inference_batch={inference_batch_size}；数据："
        "games/iter={games_per_iteration}，buffer={replay_buffer_size:,}；"
        "训练：batch={training_batch_size}，"
        "steps/iter={train_steps_per_iteration}，lr={learning_rate}",
        "探索：playouts={playouts}、c_puct={c_puct}、"
        "inference_batch={inference_batch_size}；data："
        "games/iter={games_per_iteration}、buffer={replay_buffer_size:,}；"
        "training：batch={training_batch_size}、"
        "steps/iter={train_steps_per_iteration}、lr={learning_rate}",
    ),
    "startup.schedule": _translations(
        "Run: iterations={iterations}, device={device}, model={model_preset}; "
        "short game: start={start_position_mode}, mode={adjudication_mode}, "
        "max={max_moves}, N={capture_threshold}; noise: "
        "alpha={dirichlet_alpha}, weight={dirichlet_weight}; evaluation: every "
        "{check_frequency} iterations, {evaluation_games} games, "
        "promotion threshold={promotion_score:.1%}",
        "运行：iterations={iterations}，device={device}，model={model_preset}；"
        "短局：start={start_position_mode}，mode={adjudication_mode}，"
        "max={max_moves}，N={capture_threshold}；噪声："
        "alpha={dirichlet_alpha}，weight={dirichlet_weight}；检查：每 "
        "{check_frequency} 轮，{evaluation_games} 局，"
        "晋级阈值={promotion_score:.1%}",
        "実行：iterations={iterations}、device={device}、model={model_preset}；"
        "短局：start={start_position_mode}、mode={adjudication_mode}、"
        "max={max_moves}、N={capture_threshold}；noise："
        "alpha={dirichlet_alpha}、weight={dirichlet_weight}；評価："
        "{check_frequency} iterations ごと、{evaluation_games} 局、"
        "昇格しきい値={promotion_score:.1%}",
    ),
    "startup.config-changed": _translations(
        "Detected {changes_count} configuration changes while resuming.",
        "恢复时检测到 {changes_count} 项配置变化。",
        "再開時に {changes_count} 件の設定変更を検出しました。",
    ),
    "startup.full-config": _translations(
        "{config_json}",
        "{config_json}",
        "{config_json}",
    ),
    "checkpoint.reference-saved": _translations(
        "Frozen iteration-0 reference model: {checkpoint}",
        "已冻结 iteration-0 参考模型：{checkpoint}",
        "iteration-0 参照モデルを固定しました：{checkpoint}",
    ),
    "evaluation.tactical-benchmark": _translations(
        "Tactical benchmark ({label}): top1={capture_top1_accuracy:.1%}, "
        "capture probability={capture_probability:.3f}, "
        "change from initial={capture_probability_improvement:+.3f}, "
        "positive value rate={positive_value_rate:.1%}",
        "战术 benchmark（{label}）：top1={capture_top1_accuracy:.1%}，"
        "吃子概率={capture_probability:.3f}，"
        "较初始={capture_probability_improvement:+.3f}，"
        "value 正值率={positive_value_rate:.1%}",
        "戦術 benchmark（{label}）：top1={capture_top1_accuracy:.1%}、"
        "捕獲確率={capture_probability:.3f}、"
        "初期比={capture_probability_improvement:+.3f}、"
        "value 正値率={positive_value_rate:.1%}",
    ),
    "evaluation.reference-evaluation-start": _translations(
        "Starting evaluation against the iteration-0 reference: {reference_games} games",
        "开始对 iteration-0 参考模型评估：{reference_games} 局",
        "iteration-0 参照モデルとの評価を開始：{reference_games} 局",
    ),
    "evaluation.reference-heartbeat": _translations(
        "Reference evaluation game {game}, ply {ply}: searching",
        "参考评估第 {game} 局、第 {ply} 手搜索中",
        "参照評価第 {game} 局、{ply} 手目を探索中",
    ),
    "evaluation.reference-game": _translations(
        "Reference evaluation {game}/{games}: current model score {score:.3f}",
        "参考评估 {game}/{games}：当前模型得分 {score:.3f}",
        "参照評価 {game}/{games}：現在モデルの score {score:.3f}",
    ),
    "evaluation.reference-evaluation-complete": _translations(
        "Score against iteration-0={score:.3f}; {diagnostic_text}",
        "对 iteration-0 得分={score:.3f}；{diagnostic_text}",
        "iteration-0 に対する score={score:.3f}；{diagnostic_text}",
    ),
    "checkpoint.best-saved": _translations(
        "Best model saved and published: {checkpoint}",
        "最佳模型已保存并发布：{checkpoint}",
        "best model を保存して公開しました：{checkpoint}",
    ),
    "checkpoint.checkpoint": _translations(
        "Training state saved: reason={reason}, stage={current_stage}",
        "训练状态已保存：reason={reason}，stage={current_stage}",
        "トレーニング状態を保存：reason={reason}、stage={current_stage}",
    ),
    "error.checkpoint-failed": _translations(
        "Could not save training state: {detail}",
        "保存训练状态失败：{detail}",
        "トレーニング状態を保存できません：{detail}",
    ),
    "training.training-heartbeat": _translations(
        "Network training is still running: step {step}/{total_steps}",
        "网络训练仍在运行：step {step}/{total_steps}",
        "ネットワーク学習を実行中：step {step}/{total_steps}",
    ),
    "checkpoint.checkpoint-retried": _translations(
        "Recovered from checkpoint file contention; write succeeded on "
        "attempt {attempts}.",
        "checkpoint 文件占用已恢复：第 {attempts} 次写入成功。",
        "checkpoint ファイル競合から復旧し、{attempts} 回目で書き込みに成功しました。",
    ),
}


_WINNER_TEXT = {
    TrainingLanguage.ENGLISH: {
        "draw": "draw",
        "black": "Black wins",
        "white": "White wins",
    },
    TrainingLanguage.CHINESE: {
        "draw": "和棋",
        "black": "先手胜",
        "white": "后手胜",
    },
    TrainingLanguage.JAPANESE: {
        "draw": "引き分け",
        "black": "先手勝ち",
        "white": "後手勝ち",
    },
}

_PROMOTION_TEXT = {
    TrainingLanguage.ENGLISH: {True: "model promoted", False: "model not promoted"},
    TrainingLanguage.CHINESE: {True: "模型已晋级", False: "模型未晋级"},
    TrainingLanguage.JAPANESE: {True: "モデル昇格", False: "モデル昇格なし"},
}

_DIAGNOSTIC_TEXT = {
    TrainingLanguage.ENGLISH: {True: "diagnostic passed", False: "diagnostic failed"},
    TrainingLanguage.CHINESE: {True: "诊断通过", False: "诊断未通过"},
    TrainingLanguage.JAPANESE: {True: "診断合格", False: "診断不合格"},
}

_ITERATION_OUTCOME = {
    TrainingLanguage.ENGLISH: {
        "promoted": "model promoted",
        "not_promoted": "model not promoted",
        "not_evaluated": "not evaluated this iteration",
    },
    TrainingLanguage.CHINESE: {
        "promoted": "已晋级",
        "not_promoted": "未晋级",
        "not_evaluated": "本轮未评估",
    },
    TrainingLanguage.JAPANESE: {
        "promoted": "昇格",
        "not_promoted": "昇格なし",
        "not_evaluated": "この iteration では未評価",
    },
}


def default_message_key(stage: str, event: str) -> str:
    return f"{stage}.{event}"


def render_training_message(
    message_key: str,
    language: str | TrainingLanguage,
    fields: Mapping[str, Any],
) -> str:
    selected = TrainingLanguage.parse(language)
    try:
        template = MESSAGE_CATALOG[message_key][selected]
    except KeyError as exc:
        raise KeyError(
            f"Unknown training message key or language: {message_key!r}, "
            f"{selected.value!r}"
        ) from exc
    values = dict(fields)
    values["winner_text"] = _WINNER_TEXT[selected][_winner_key(values.get("winner"))]
    values["promotion_text"] = _PROMOTION_TEXT[selected][
        bool(values.get("promoted", False))
    ]
    values["diagnostic_text"] = _DIAGNOSTIC_TEXT[selected][
        bool(values.get("diagnostic_passed", False))
    ]
    if "eta_seconds" in values:
        values["eta"] = format_duration(float(values["eta_seconds"]))
    if "seconds" in values:
        values["elapsed"] = format_duration(float(values["seconds"]))
    terminal_counts = values.get("terminal_counts", {})
    values["terminal_counts_text"] = json.dumps(
        terminal_counts,
        ensure_ascii=False,
        sort_keys=True,
    )
    loss = values.get("loss")
    values["loss_text"] = "n/a" if loss is None else f"{float(loss):.4f}"
    evaluated = bool(values.get("evaluated", False))
    if not evaluated:
        outcome = "not_evaluated"
    else:
        outcome = "promoted" if values.get("promoted") else "not_promoted"
    values["iteration_outcome"] = _ITERATION_OUTCOME[selected][outcome]
    try:
        return template.format_map(values)
    except KeyError as exc:
        raise ValueError(
            f"Message {message_key!r} is missing field {exc.args[0]!r}"
        ) from exc


def validate_message_catalog() -> None:
    expected_languages = set(TrainingLanguage)
    formatter = Formatter()
    for key, translations in MESSAGE_CATALOG.items():
        if set(translations) != expected_languages:
            missing = expected_languages - set(translations)
            raise ValueError(
                f"Training message {key!r} is missing languages: "
                f"{sorted(item.value for item in missing)}"
            )
        placeholders = {
            language: {
                field_name.split(".", 1)[0].split("[", 1)[0]
                for _, field_name, _, _ in formatter.parse(template)
                if field_name
            }
            for language, template in translations.items()
        }
        expected = placeholders[TrainingLanguage.ENGLISH]
        for language, names in placeholders.items():
            if names != expected:
                raise ValueError(
                    f"Training message {key!r} has inconsistent placeholders "
                    f"for {language.value}: expected {sorted(expected)}, "
                    f"got {sorted(names)}"
                )


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(rounded, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{remaining_minutes:02d}m{remaining_seconds:02d}s"
    if remaining_minutes:
        return f"{remaining_minutes:d}m{remaining_seconds:02d}s"
    return f"{remaining_seconds:d}s"


def _winner_key(winner: Any) -> str:
    if winner is None:
        return "draw"
    return "black" if int(winner) == 0 else "white"


validate_message_catalog()
