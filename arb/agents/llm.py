"""LLM recovery controller: ``policy="llm_based"``.

The model sits exactly where the rule table and the learned ranker sit -- at the
operator-selection step. The detector, the environment, the recovery programs and
the grader are unchanged, so a difference from the rule baseline is attributable
to *which* operator gets applied, not to different capabilities.

Two properties make this an evaluation rather than a demo:

* **No oracle.** The prompt carries only the monitor's view: the detector's
  posterior over failure classes and the monitor feature vector. It never sees
  the injected fault, its injection step, the expected state, or the grader.
* **Replayable.** Every reply is cached on disk under a hash of the exact prompt,
  so a scored run can be reproduced bit for bit and re-runs cost nothing.

The backend is any Ollama server (``/api/chat``); ``--base-url`` points at a
remote one. No credential is involved, and the model is chosen on the command
line, so the release ships a protocol rather than a fixed vendor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from arb.agents.policies import Decision
from arb.perturb import CANONICAL_PROGRAM, program_for
from arb.taxonomy import RecoveryAction

#: One line per operator: what the model is choosing between.
ACTION_MENU: dict[RecoveryAction, str] = {
    RecoveryAction.CONTINUE: "alarm is a false positive; keep executing the plan unchanged",
    RecoveryAction.RETRY: "re-issue the identical call once (transient transport error only)",
    RecoveryAction.REVALIDATE_ARGS: "repair the call's arguments against the tool schema, then re-issue",
    RecoveryAction.REFRESH_STATE: "re-read the object being mutated so the write uses current state",
    RecoveryAction.RE_READ_AUTHORITATIVE: "re-read the authoritative source and adopt its value over other observations",
    RecoveryAction.REPLAN: "discard the remaining steps and recompute the plan from current state",
    RecoveryAction.RESYNC_MESSAGE: "re-send or re-read the inter-agent message so the recipient acts on the sender's plan",
    RecoveryAction.REANCHOR_OBJECTIVE: "restore the stated objective and recompute the plan under it",
    RecoveryAction.ESCALATE_ABORT: "halt cleanly with no further mutations and escalate to a human",
}

SYSTEM_PROMPT = (
    "You are the recovery controller of a tool-using agent. A monitor has raised "
    "an alarm after a tool call. Choose exactly one recovery operator from the "
    "menu: the one whose precondition matches the evidence. Prefer the least "
    "disruptive operator that addresses the cause; do not invent operators. "
    'Reply with JSON only: {"action": "<operator>", "confidence": <0-1>, '
    '"reason": "<10 words max>"}.'
)


@dataclass
class LLMConfig:
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    max_tokens: int = 200
    timeout: float = 240.0
    cache_path: str = "results/llm_cache.jsonl"


@dataclass
class LLMClient:
    """Ollama chat client with an append-only disk cache and usage accounting."""

    config: LLMConfig = field(default_factory=LLMConfig)
    cache: dict[str, str] = field(default_factory=dict)
    usage: dict[str, float] = field(
        default_factory=lambda: {"calls": 0, "cache_hits": 0, "tokens_in": 0,
                                 "tokens_out": 0, "seconds": 0.0}
    )
    #: Episodes are evaluated in a thread pool; the cache and counters are shared.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if os.path.exists(self.config.cache_path):
            with open(self.config.cache_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row["key"]] = row["reply"]

    # ---- prompt -> reply -------------------------------------------------- #
    def complete(self, user: str) -> str:
        key = hashlib.sha256(
            f"{self.config.model}\x00{SYSTEM_PROMPT}\x00{user}".encode("utf-8")
        ).hexdigest()
        with self.lock:
            if key in self.cache:
                self.usage["cache_hits"] += 1
                return self.cache[key]

        body = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.config.temperature, "seed": 0,
                        "num_predict": self.config.max_tokens},
        }
        started = time.time()
        try:
            req = urllib.request.Request(
                f"{self.config.base_url.rstrip('/')}/api/chat",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"LLM backend unreachable at {self.config.base_url} "
                f"(start `ollama serve` and `ollama pull {self.config.model}`): {exc}"
            ) from exc

        reply = payload["message"]["content"]
        with self.lock:
            self.usage["calls"] += 1
            self.usage["seconds"] += time.time() - started
            self.usage["tokens_in"] += int(payload.get("prompt_eval_count") or 0)
            self.usage["tokens_out"] += int(payload.get("eval_count") or 0)
            self.cache[key] = reply
            os.makedirs(os.path.dirname(self.config.cache_path) or ".", exist_ok=True)
            with open(self.config.cache_path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps({"key": key, "model": self.config.model,
                                     "reply": reply}) + "\n")
        return reply


def _prompt(*, detected_type: str, posterior: dict[str, float], features: dict[str, float],
            step_index: int, retries_used: int, max_retries: int) -> str:
    ranked = sorted(posterior.items(), key=lambda kv: -kv[1])[:4] or [(detected_type, 1.0)]
    evidence = sorted(((k, float(v)) for k, v in features.items() if v), key=lambda kv: -kv[1])
    menu = "\n".join(f"- {a.value}: {text}" for a, text in ACTION_MENU.items())
    return (
        f"Alarm at step {step_index}. Retries used {retries_used}/{max_retries}.\n"
        f"Failure posterior: {', '.join(f'{k}={v:.3f}' for k, v in ranked)}\n"
        f"Monitor evidence: {', '.join(f'{k}={v:.2f}' for k, v in evidence[:12]) or 'none'}\n"
        f"Operators:\n{menu}\n"
        "Choose one operator."
    )


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def _parse_action(reply: str) -> tuple[RecoveryAction | None, float, str]:
    """Tolerate prose and code fences; reject anything outside the menu."""
    match = _JSON_OBJECT.search(reply or "")
    if not match:
        return None, 0.0, "unparseable"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 0.0, "unparseable"
    try:
        action = RecoveryAction(str(obj.get("action", "")).strip().lower())
    except ValueError:
        return None, 0.0, f"off-menu:{obj.get('action')!r}"
    try:
        confidence = min(1.0, max(0.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return action, confidence, str(obj.get("reason", ""))[:120]


class LLMRecoveryPolicy:
    """Operator selection by an LLM over the shared action space."""

    name = "llm_based"
    on_error = False

    def __init__(self, client: LLMClient | None = None, max_retries: int = 2) -> None:
        self.client = client or LLMClient()
        self.max_retries = max_retries
        #: Per-run diagnostics, kept on the policy so a caller can report them.
        self.actions: dict[str, int] = {}
        self.fallbacks = 0

    def reset(self) -> None:
        """Nothing to clear: usage belongs to the shared client, not the episode."""
        return None

    def decide(
        self,
        *,
        detected_type: str | None = None,
        posterior: dict[str, float] | None = None,
        step_index: int = 0,
        injections_handled: int = 0,
        retries_used: int = 0,
        features: dict[str, float] | None = None,
        **kw: Any,
    ) -> Decision:
        if detected_type is None:
            return Decision(RecoveryAction.CONTINUE, (RecoveryAction.CONTINUE,), 0.3,
                            "no detection")
        prompt = _prompt(detected_type=detected_type,
                         posterior=posterior or {detected_type: 1.0},
                         features=features or {}, step_index=step_index,
                         retries_used=retries_used, max_retries=self.max_retries)
        action, confidence, reason = _parse_action(self.client.complete(prompt))
        if action is None:
            # A reply outside the menu is not a decision: fall back to the
            # canonical repair and count it, rather than guessing an operator.
            with self.client.lock:
                self.fallbacks += 1
            action = CANONICAL_PROGRAM[detected_type][0]
            confidence, reason = 0.0, f"fallback ({reason})"
        if action is RecoveryAction.RETRY and retries_used >= self.max_retries:
            action = CANONICAL_PROGRAM[detected_type][1]
            reason = f"retry budget exhausted -> {action.value}"

        with self.client.lock:
            self.actions[action.value] = self.actions.get(action.value, 0) + 1
        return Decision(action, program_for(action), confidence,
                        f"llm choice={action.value} ({reason})")


__all__ = ["ACTION_MENU", "LLMClient", "LLMConfig", "LLMRecoveryPolicy"]
