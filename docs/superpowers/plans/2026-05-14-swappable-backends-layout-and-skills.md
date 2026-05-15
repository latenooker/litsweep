# Swappable Backends, Layout Cleanup, and Deploy Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make litsweep shareable with non-Stanford colleagues by adding swappable embed/label backends (Stanford + Ollama at v1), tidying the per-project `results/` layout, and shipping installable Claude + Codex skills.

**Architecture:** Three independent slices (each its own PR): (1) a tiny registry in `label_backends.py` / `embed_backends.py` at the repo root; drivers in `scripts/` shrink to argparse + loop. (2) Additive subdirectories under `results/` (`gapfills/`, `pilots/`, `analysis/`, `archive/`, `logs/`); canonical top-level filenames unchanged so cross-project tools keep working. (3) Symlink-installable skills for `~/.claude/skills/` and `~/.codex/skills/`.

**Tech Stack:** Python 3.11+, `requests`, `urllib.request`, `pandas`, `numpy`, `pytest`, `unittest.mock`. No new runtime deps.

**Reference:** [`docs/superpowers/specs/2026-05-14-swappable-backends-and-skills-design.md`](../specs/2026-05-14-swappable-backends-and-skills-design.md)

---

## File map

**Slice 1 — Backend registry**
- Create: `label_backends.py`, `embed_backends.py`, `tests/test_label_backends.py`, `tests/test_embed_backends.py`
- Modify: `scripts/label_with_stanford.py`, `scripts/embed_filter.py`, `scripts/scaffold_new_search.py`

**Slice 2 — Layout cleanup**
- Create: `scripts/disk_hygiene.py`, `scripts/migrate_layout.py`, `tests/test_migrate_layout.py`, `tests/test_scaffold_smoke.py`
- Modify: `scripts/scaffold_new_search.py`, `litsweep_search.py`, `scripts/embed_filter.py`, `scripts/label_with_stanford.py`, `docs/DIRECTORY_STRUCTURE.md`, `docs/DISK_HYGIENE.md`, `docs/DEPLOYING_A_NEW_SEARCH.md`

**Slice 3 — Skills**
- Create: `skills/claude/litsweep-deploy/SKILL.md`, `skills/codex/litsweep-deploy/SKILL.md`, `scripts/install_skills.sh`
- Modify: `README.md`

---

# Slice 1: Backend registry

## Task 1: Add `BackendError` and label-backend registry skeleton

**Files:**
- Create: `label_backends.py`
- Create: `tests/test_label_backends.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_label_backends.py`:

```python
"""Tests for label_backends registry dispatch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import label_backends


def test_make_backend_unknown_name_exits():
    with pytest.raises(SystemExit) as exc:
        label_backends.make_backend("does_not_exist", api_key="x")
    assert "does_not_exist" in str(exc.value)
    assert "stanford" in str(exc.value)  # lists valid options


def test_backend_error_carries_reason():
    err = label_backends.BackendError("HTTP 500")
    assert err.reason == "HTTP 500"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_label_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'label_backends'`.

- [ ] **Step 3: Write minimal implementation**

Create `label_backends.py`:

```python
"""Label-backend registry: pick which LLM provider drives `label_with_stanford.py`.

V1 ships Stanford gateway (existing behavior, default) and local Ollama
(`/api/chat` with `format="json"`). Additional providers (OpenAI,
Anthropic, Voyage, etc.) slot in as one new dataclass + one entry in
`BACKENDS`.

The driver in `scripts/label_with_stanford.py` keeps the project's
`SYSTEM_PROMPT` and `_error_label` (both tied to per-project schemas).
Backends only know how to call their provider and return a parsed dict
on success, or raise `BackendError` on permanent failure.
"""

from __future__ import annotations

from typing import Any


class BackendError(Exception):
    """Permanent failure from a label backend (retries exhausted, bad config).

    The driver converts ``reason`` into a project-shaped error label via
    its local ``_error_label`` so the schema dependency stays in the
    driver, not the backend.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


BACKENDS: dict[str, type] = {}


def make_backend(name: str, **kwargs: Any):
    """Build a backend by name; raise SystemExit on unknown name."""
    if name not in BACKENDS:
        raise SystemExit(
            f"Unknown label backend: {name!r}. "
            f"Valid: {sorted(BACKENDS)}"
        )
    cls = BACKENDS[name]
    accepted = {k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__}
    return cls(**accepted)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_label_backends.py -v`
Expected: `test_backend_error_carries_reason PASSED`; `test_make_backend_unknown_name_exits PASSED`.

- [ ] **Step 5: Commit**

```bash
git add label_backends.py tests/test_label_backends.py
git commit -m "$(cat <<'EOF'
label_backends: registry skeleton + BackendError

Empty BACKENDS dict + make_backend() dispatch with friendly error on
unknown names. StanfordBackend and OllamaLabelBackend land in
follow-up commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Port `StanfordBackend` into the registry

**Files:**
- Modify: `label_backends.py`
- Modify: `tests/test_label_backends.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_label_backends.py`:

```python
import json
from unittest.mock import patch, MagicMock


def _mock_urlopen_ok(payload: dict):
    """Build a context-manager mock that returns ``payload`` JSON."""
    cm = MagicMock()
    cm.__enter__.return_value = MagicMock(
        read=MagicMock(return_value=json.dumps(payload).encode("utf-8"))
    )
    cm.__exit__.return_value = False
    return cm


def test_stanford_backend_parses_json_response():
    backend = label_backends.make_backend("stanford", api_key="sk-x")
    payload = {
        "choices": [
            {"message": {"content": '{"relevance": "core", "rationale": "ok"}'}}
        ]
    }
    with patch("label_backends.urllib.request.urlopen",
               return_value=_mock_urlopen_ok(payload)):
        result = backend.call("prompt", "system")
    assert result == {"relevance": "core", "rationale": "ok"}


def test_stanford_backend_strips_code_fences():
    backend = label_backends.make_backend("stanford", api_key="sk-x")
    payload = {
        "choices": [{"message": {"content": '```json\n{"relevance": "core"}\n```'}}]
    }
    with patch("label_backends.urllib.request.urlopen",
               return_value=_mock_urlopen_ok(payload)):
        result = backend.call("prompt", "system")
    assert result == {"relevance": "core"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_label_backends.py -v -k stanford`
Expected: FAIL — `Unknown label backend: 'stanford'`.

- [ ] **Step 3: Write the implementation**

Edit `label_backends.py`. Replace the module docstring's last paragraph and the imports + add the Stanford backend. Top of file becomes:

```python
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("label_backends")


class BackendError(Exception):
    """Permanent failure from a label backend (retries exhausted, bad config)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3
MAX_429_RETRIES = 8


def _strip_code_fence(text: str) -> str:
    """Drop ```json ... ``` fences if a model added them despite instructions."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


@dataclass
class StanfordBackend:
    """Stanford AI API gateway (OpenAI-compatible chat-completions).

    Carries over from the pre-registry ``call_stanford`` verbatim:
    bearer-token auth, 429-aware retry loop with bounded backoff,
    optional client-side rate limiter via ``min_interval_s``.

    Attributes:
        api_key: Required. Pass ``STANFORD_API_KEY`` here.
        api_url: Base URL; default Stanford prod.
        model: Model id; default ``gemini-2.0-flash-lite-001``.
        temperature, max_tokens: Forwarded to the chat endpoint.
        min_interval_s: Minimum wall-clock seconds between calls (>0 to
            pre-throttle below the gateway's quota; see notes in
            ``label_with_stanford.py`` --min-interval-s docstring).
    """

    api_key: str
    api_url: str = "https://aiapi-prod.stanford.edu/v1"
    model: str = "gemini-2.0-flash-lite-001"
    temperature: float = 0.0
    max_tokens: int = 400
    min_interval_s: float = 0.0
    _last_request_t: float = field(default=0.0, init=False, repr=False)

    def check(self) -> None:
        if not self.api_key:
            raise SystemExit(
                "Stanford API key not set. Pass --api-key or STANFORD_API_KEY."
            )
        url = f"{self.api_url.rstrip('/')}/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
            models = [m.get("id") for m in body.get("data", []) if m.get("id")]
            logger.info("Stanford reachable. %d models available.", len(models))
            if self.model not in models:
                logger.warning(
                    "Model %r not in list; will attempt anyway. Sample: %s",
                    self.model, models[:5],
                )
        except Exception as exc:  # pragma: no cover - probe is best-effort
            logger.warning("Could not list Stanford models (probe only): %s", exc)

    def call(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        if self.min_interval_s > 0:
            elapsed = time.monotonic() - self._last_request_t
            if elapsed < self.min_interval_s:
                time.sleep(self.min_interval_s - elapsed)
        self._last_request_t = time.monotonic()

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")

        url = f"{self.api_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        attempt = 0
        rate_limit_attempts = 0
        while attempt < MAX_RETRIES:
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                    body = json.loads(resp.read())
                content = body["choices"][0]["message"]["content"]
                return json.loads(_strip_code_fence(content))
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
                if exc.code == 429:
                    rate_limit_attempts += 1
                    if rate_limit_attempts >= MAX_429_RETRIES:
                        raise BackendError(
                            f"HTTP 429 after {MAX_429_RETRIES} retries"
                        )
                    wait = min(2 ** (rate_limit_attempts + 1), 60)
                    logger.warning("429 rate limit (%d/%d), sleeping %ds",
                                   rate_limit_attempts, MAX_429_RETRIES, wait)
                    time.sleep(wait)
                    continue
                elif attempt < MAX_RETRIES - 1:
                    logger.warning("HTTP %d (attempt %d): %s",
                                   exc.code, attempt + 1, err_body[:200])
                    time.sleep(2 ** attempt)
                else:
                    raise BackendError(f"HTTP {exc.code}: {err_body[:200]}")
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                if attempt < MAX_RETRIES - 1:
                    logger.warning("parse error attempt %d: %s", attempt + 1, exc)
                    time.sleep(1)
                else:
                    raise BackendError(f"parse: {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                if attempt < MAX_RETRIES - 1:
                    logger.warning("attempt %d failed: %s", attempt + 1, exc)
                    time.sleep(2 ** attempt)
                else:
                    raise BackendError(str(exc))
            attempt += 1
        raise BackendError("max retries exhausted")


BACKENDS: dict[str, type] = {
    "stanford": StanfordBackend,
}


def make_backend(name: str, **kwargs: Any):
    """Build a backend by name; raise SystemExit on unknown name."""
    if name not in BACKENDS:
        raise SystemExit(
            f"Unknown label backend: {name!r}. "
            f"Valid: {sorted(BACKENDS)}"
        )
    cls = BACKENDS[name]
    accepted = {k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__}
    return cls(**accepted)
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_label_backends.py -v`
Expected: 4 passed (the original two + the two new Stanford tests).

- [ ] **Step 5: Commit**

```bash
git add label_backends.py tests/test_label_backends.py
git commit -m "$(cat <<'EOF'
label_backends: StanfordBackend ported verbatim into registry

Carries the existing call_stanford / check_stanford / 429-retry logic
into a dataclass. Raises BackendError on permanent failure instead of
returning the error-label dict — that translation stays in the driver
where SYSTEM_PROMPT and _error_label live.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `OllamaLabelBackend`

**Files:**
- Modify: `label_backends.py`
- Modify: `tests/test_label_backends.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_label_backends.py`:

```python
def _mock_requests_post(json_payload: dict, status: int = 200):
    """Mock for requests.post returning ``json_payload``."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_payload
    resp.raise_for_status.side_effect = (
        None if status < 400 else Exception(f"HTTP {status}")
    )
    return resp


def test_ollama_label_backend_parses_chat_response():
    backend = label_backends.make_backend("ollama")
    payload = {
        "message": {"content": '{"relevance": "adjacent", "rationale": "x"}'}
    }
    with patch("label_backends.requests.post",
               return_value=_mock_requests_post(payload)):
        result = backend.call("prompt", "system")
    assert result == {"relevance": "adjacent", "rationale": "x"}


def test_ollama_label_backend_raises_on_unparseable():
    backend = label_backends.make_backend("ollama")
    payload = {"message": {"content": "not json at all"}}
    with patch("label_backends.requests.post",
               return_value=_mock_requests_post(payload)):
        with pytest.raises(label_backends.BackendError):
            backend.call("prompt", "system")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_label_backends.py -v -k ollama`
Expected: FAIL — `Unknown label backend: 'ollama'`.

- [ ] **Step 3: Write the implementation**

Edit `label_backends.py`. Add `import requests` near the top. Add the new dataclass before the `BACKENDS` dict and register it:

```python
import requests  # add to imports
```

Add before `BACKENDS`:

```python
@dataclass
class OllamaLabelBackend:
    """Local Ollama chat backend with ``format="json"`` for structured labels.

    Reuses the same daemon the embedder already needs. Smaller open-weights
    models are less reliable at strict-JSON than gateway gemini/gpt — set
    ``model`` to something instruction-tuned (llama3.1, qwen2.5, mistral-nemo)
    and verify with a 50-record pilot before a full run.

    Attributes:
        host: Ollama base URL.
        model: Ollama model tag (must be pulled: ``ollama pull <model>``).
        temperature: Forwarded; 0.0 for label determinism.
        num_predict: Max tokens; Ollama's analogue of max_tokens.
        timeout_s: Per-request timeout.
        max_retries: Retries on RequestException or unparseable JSON.
    """

    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b-instruct-q4_K_M"
    temperature: float = 0.0
    num_predict: int = 400
    timeout_s: int = 300
    max_retries: int = 3

    def check(self) -> None:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(
                f"Cannot reach Ollama at {self.host}: {exc}\n"
                f"Start it with: ollama serve   "
                f"(and ensure: ollama pull {self.model})"
            )
        tags = [m["name"] for m in resp.json().get("models", [])]
        if not any(t.startswith(self.model.split(":")[0]) for t in tags):
            raise SystemExit(
                f"Model '{self.model}' not pulled. Run: ollama pull {self.model}"
            )

    def call(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        url = f"{self.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                return json.loads(_strip_code_fence(content))
            except (requests.RequestException, ValueError, KeyError) as err:
                last_err = err
                logger.warning(
                    "Ollama label attempt %d/%d failed: %s",
                    attempt, self.max_retries, err,
                )
                time.sleep(1.5 * attempt)
        raise BackendError(f"Ollama label failed after {self.max_retries} retries: {last_err}")
```

Register it in `BACKENDS`:

```python
BACKENDS: dict[str, type] = {
    "stanford": StanfordBackend,
    "ollama":   OllamaLabelBackend,
}
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_label_backends.py -v`
Expected: 6 passed (original 4 + 2 new Ollama tests).

- [ ] **Step 5: Commit**

```bash
git add label_backends.py tests/test_label_backends.py
git commit -m "$(cat <<'EOF'
label_backends: add OllamaLabelBackend

Local Ollama /api/chat with format="json" for structured labels. Same
retry shape as Stanford; no 429 handling (local). Default model
llama3.1:8b-instruct-q4_K_M — override with --ollama-label-model.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Refactor `label_with_stanford.py` driver to use the registry

**Files:**
- Modify: `scripts/label_with_stanford.py`

- [ ] **Step 1: Re-read the existing file**

Before editing, read `scripts/label_with_stanford.py` so the diff stays surgical. Lines 60–298 currently hold `StanfordConfig`, `check_stanford`, `_strip_code_fence`, `_error_label`, `call_stanford`, and the module-level `_LAST_REQUEST_T`. Those move out (or get deleted, having moved into `label_backends.py`). `_error_label` and `_row_prompt` stay (project-schema-tied and prompt-building, respectively).

- [ ] **Step 2: Delete the moved code, import the registry, keep `_error_label`**

Replace the block from `STANFORD_DEFAULT_URL = ...` (line ~60) through the end of `call_stanford` (line ~298) with:

```python
# Make label_backends importable when running this script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import label_backends  # noqa: E402

STANFORD_DEFAULT_URL = "https://aiapi-prod.stanford.edu/v1"
STANFORD_DEFAULT_MODEL = "gemini-2.0-flash-lite-001"
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"


def _error_label(reason: str) -> dict[str, Any]:
    """Project-schema error label. Tied to SYSTEM_PROMPT — stays in the driver."""
    return {
        "relevance": "error",
        "parent_lithology": "error",
        "process_focus": "error",
        "regolith_zone": "error",
        "methods": [],
        "minerals": [],
        "climate_zone": "not_applicable",
        "is_thesis": False,
        "is_review": False,
        "rationale": reason[:200],
    }
```

- [ ] **Step 3: Replace the `call_stanford(prompt, cfg)` call site in the main loop**

In the `for _, row in pbar:` block, change:

```python
result = call_stanford(prompt, cfg)
```

to:

```python
try:
    result = backend.call(prompt, SYSTEM_PROMPT)
except label_backends.BackendError as exc:
    result = _error_label(exc.reason)
```

- [ ] **Step 4: Replace the argparse + config construction**

Replace the existing Stanford-only argparse block (around lines 379–423) and the `cfg = StanfordConfig(...)` / `check_stanford(cfg)` lines with:

```python
parser.add_argument(
    "--label-backend", choices=sorted(label_backends.BACKENDS),
    default="stanford",
    help="Which label backend to use. stanford (default) needs "
         "STANFORD_API_KEY; ollama needs a local Ollama daemon with the "
         "chat model pulled.",
)
parser.add_argument("--api-key", default=os.environ.get("STANFORD_API_KEY", ""),
                    help="Stanford API key (--label-backend stanford only).")
parser.add_argument("--api-url", default=STANFORD_DEFAULT_URL,
                    help="Stanford base URL (--label-backend stanford only).")
parser.add_argument("--model", default=None,
                    help="Model id. Default per-backend: "
                         f"stanford={STANFORD_DEFAULT_MODEL}, "
                         f"ollama={OLLAMA_DEFAULT_MODEL}.")
parser.add_argument("--ollama-host", default=OLLAMA_DEFAULT_HOST,
                    help="Ollama base URL (--label-backend ollama only).")
parser.add_argument(
    "--min-interval-s", type=float, default=0.0,
    help="Pre-throttle: minimum seconds between successive gateway "
         "requests. Default 0 (reactive only — back off on 429). "
         "stanford backend only.",
)
# ... existing flags below this point (--checkpoint-dir, --chunk-size, etc.) unchanged.
```

Then build the backend after parsing:

```python
backend_kwargs = {
    "api_key":         args.api_key,
    "api_url":         args.api_url,
    "host":            args.ollama_host,
    "min_interval_s":  args.min_interval_s,
}
if args.model:
    backend_kwargs["model"] = args.model
elif args.label_backend == "ollama":
    backend_kwargs["model"] = OLLAMA_DEFAULT_MODEL
# stanford's dataclass default already matches STANFORD_DEFAULT_MODEL.

backend = label_backends.make_backend(args.label_backend, **backend_kwargs)
backend.check()
```

- [ ] **Step 5: Delete checkpoints after a successful merged-CSV write**

Spec §6: checkpoints are crash insurance, not deliverables — once the
merged corpus writes, they've served their purpose. A failed write
leaves them in place so a resume works.

In `main()`, immediately after the existing `merged.to_csv(args.out, index=False)`
line and its success log, add:

```python
import shutil  # add to imports at top of file if not present

if checkpoint_dir.exists():
    shutil.rmtree(checkpoint_dir)
    logger.info("removed checkpoints %s (corpus write succeeded)", checkpoint_dir)
```

Place it after the `logger.info("wrote %s ...")` call so an exception
during `to_csv` leaves the checkpoint dir intact.

- [ ] **Step 6: Remove the now-unused `call_stanford` / `check_stanford` / `_strip_code_fence` / `StanfordConfig` definitions if any remain**

Run: `grep -n "call_stanford\|check_stanford\|StanfordConfig\|_strip_code_fence\|_LAST_REQUEST_T" scripts/label_with_stanford.py`
Expected: no matches.

- [ ] **Step 7: Smoke-test with `--dry-run`-equivalent (no API call)**

The script doesn't have `--dry-run`, but the smallest possible invocation that exercises argparse + backend construction is:

```bash
python scripts/label_with_stanford.py \
    --csv /dev/null --out /tmp/out.csv \
    --label-backend ollama --ollama-host http://127.0.0.1:1
```

Expected: SystemExit with `Cannot reach Ollama at http://127.0.0.1:1` (the check() ran). This confirms argparse and registry dispatch work without needing a real Stanford key or real Ollama.

- [ ] **Step 8: Commit**

```bash
git add scripts/label_with_stanford.py
git commit -m "$(cat <<'EOF'
label_with_stanford: shrink to driver, dispatch through label_backends

--label-backend stanford (default) preserves existing CLI; --label-backend
ollama routes to local Ollama. SYSTEM_PROMPT and _error_label stay in
the driver (project-schema-tied); call_stanford / check_stanford /
StanfordConfig moved into label_backends.StanfordBackend.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Create `embed_backends.py` with `OllamaEmbedBackend`

**Files:**
- Create: `embed_backends.py`
- Create: `tests/test_embed_backends.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embed_backends.py`:

```python
"""Tests for embed_backends registry dispatch and OllamaEmbedBackend."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embed_backends


def _mock_post(payload: dict):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_make_backend_unknown_name_exits():
    with pytest.raises(SystemExit) as exc:
        embed_backends.make_backend("foo")
    assert "ollama" in str(exc.value)


def test_ollama_embed_backend_returns_normalized_matrix():
    backend = embed_backends.make_backend("ollama")
    texts = ["a", "b"]
    payload = {"embeddings": [[3.0, 4.0], [0.0, 5.0]]}
    with patch("embed_backends.requests.post",
               return_value=_mock_post(payload)):
        mat = backend.embed(texts)
    assert mat.shape == (2, 2)
    norms = np.linalg.norm(mat, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embed_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'embed_backends'`.

- [ ] **Step 3: Write the implementation**

Create `embed_backends.py`:

```python
"""Embed-backend registry: pluggable embedding providers.

V1 ships local Ollama (the only existing path). The shape leaves room
for Voyage / Jina / Cohere / OpenAI-compatible additions as one new
dataclass + one entry in ``BACKENDS``.

Backends expose a single ``embed(texts) -> np.ndarray`` returning a
``(N, D)`` float32 matrix, L2-normed (so dot products are cosines).
``check()`` raises ``SystemExit`` with a friendly message if the
backend can't be reached.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger("embed_backends")


@dataclass
class OllamaEmbedBackend:
    """Local Ollama embeddings via ``/api/embed``.

    Attributes:
        model: Ollama embedding model tag. ``bge-m3`` is multilingual
            (100+ langs) and is the canonical default for the
            multilingual lit-search use case.
        host: Ollama base URL.
        batch_size: Texts per request.
        timeout_s: Per-batch timeout.
        max_retries: Retries on RequestException or bad shape.
        normalize: L2-norm output so dot product equals cosine.
    """

    model: str = "bge-m3"
    host: str = "http://localhost:11434"
    batch_size: int = 32
    timeout_s: int = 300
    max_retries: int = 3
    normalize: bool = True

    def check(self) -> None:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(
                f"Cannot reach Ollama at {self.host}: {exc}\n"
                f"Start it with: ollama serve   "
                f"(and ensure: ollama pull {self.model})"
            )
        tags = [m["name"] for m in resp.json().get("models", [])]
        if not any(t.startswith(self.model) for t in tags):
            raise SystemExit(
                f"Model '{self.model}' not pulled. Run: ollama pull {self.model}"
            )

    def _post_batch(self, texts: Sequence[str]) -> np.ndarray:
        url = f"{self.host.rstrip('/')}/api/embed"
        payload = {"model": self.model, "input": list(texts)}
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                data = resp.json()
                embeds = data.get("embeddings")
                if embeds is None and "embedding" in data:
                    embeds = [data["embedding"]]
                if not embeds:
                    raise RuntimeError(f"empty embeddings: {data!r}")
                arr = np.asarray(embeds, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] != len(texts):
                    raise RuntimeError(f"shape {arr.shape} != batch {len(texts)}")
                return arr
            except (requests.RequestException, ValueError, KeyError, RuntimeError) as err:
                last_err = err
                logger.warning(
                    "Ollama embed attempt %d/%d failed: %s",
                    attempt, self.max_retries, err,
                )
                time.sleep(1.5 * attempt)
        raise RuntimeError(
            f"Ollama embedding failed after {self.max_retries} retries: {last_err}"
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of texts; returns ``(N, D)`` float32, L2-normed."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out: list[np.ndarray] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(texts), self.batch_size),
                      total=n_batches, desc=f"embed ({self.model})"):
            batch = texts[i : i + self.batch_size]
            out.append(self._post_batch(batch))
        matrix = np.vstack(out)
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix


BACKENDS: dict[str, type] = {
    "ollama": OllamaEmbedBackend,
}


def make_backend(name: str, **kwargs: Any):
    """Build an embed backend by name; raise SystemExit on unknown name."""
    if name not in BACKENDS:
        raise SystemExit(
            f"Unknown embed backend: {name!r}. "
            f"Valid: {sorted(BACKENDS)}"
        )
    cls = BACKENDS[name]
    accepted = {k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__}
    return cls(**accepted)
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_embed_backends.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add embed_backends.py tests/test_embed_backends.py
git commit -m "$(cat <<'EOF'
embed_backends: registry skeleton + OllamaEmbedBackend

Ports the existing EmbedConfig / ping_ollama / _post_batch / embed_texts
logic into a dataclass. New backends (Voyage, Jina, OpenAI-compatible)
land later as one dataclass + one BACKENDS entry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Refactor `embed_filter.py` to use `embed_backends`

**Files:**
- Modify: `scripts/embed_filter.py`

- [ ] **Step 1: Re-read the existing file**

Locate the `EmbedConfig`, `ping_ollama`, `_post_batch`, and `embed_texts` definitions (currently lines ~100–171). Those move out. The `ANCHORS` constant, `_row_text`, cache logic, anchor encoding, score loop, and `main()` stay.

- [ ] **Step 2: Delete the moved code, import the registry**

Replace the EmbedConfig / ping_ollama / _post_batch / embed_texts block (lines ~99–171) with:

```python
# Make embed_backends importable when running this script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embed_backends  # noqa: E402
```

Remove `from dataclasses import dataclass` if it has no other use, and remove `from typing import Sequence` if unused (check with `grep`).

- [ ] **Step 3: Replace the argparse `--model` / `--host` / `--batch-size` lines and config construction**

Add `--embed-backend`:

```python
parser.add_argument(
    "--embed-backend", choices=sorted(embed_backends.BACKENDS),
    default="ollama",
    help="Which embed backend to use. Today only 'ollama'; flag exists "
         "so future Voyage/Jina/OpenAI-compatible adds slot in without "
         "breaking the CLI.",
)
parser.add_argument("--model", default="bge-m3", help="Embed model tag.")
parser.add_argument("--host", default="http://localhost:11434",
                    help="Ollama base URL (--embed-backend ollama only).")
parser.add_argument("--batch-size", type=int, default=32)
```

Replace `cfg = EmbedConfig(...)` and `ping_ollama(cfg)` with:

```python
backend = embed_backends.make_backend(
    args.embed_backend,
    model=args.model, host=args.host, batch_size=args.batch_size,
)
backend.check()
```

- [ ] **Step 4: Replace the two call sites of `embed_texts(...)`**

Find the two `embed_texts(...)` calls (the row-encoding loop and the anchor encoding). Replace with `backend.embed(...)`:

```python
chunk_mat = backend.embed(to_encode_texts[chunk_start:chunk_end])
```

```python
anchor_mat = backend.embed(ANCHORS)
```

- [ ] **Step 5: Remove the unused module-level `cfg` references**

Run: `grep -n "cfg\b\|EmbedConfig\|ping_ollama\|embed_texts\|_post_batch" scripts/embed_filter.py`
Expected: no matches (or only the import line).

- [ ] **Step 6: Smoke-test**

```bash
python scripts/embed_filter.py --csv /dev/null --out /tmp/x.csv --host http://127.0.0.1:1
```

Expected: SystemExit with `Cannot reach Ollama at http://127.0.0.1:1`.

- [ ] **Step 7: Commit**

```bash
git add scripts/embed_filter.py
git commit -m "$(cat <<'EOF'
embed_filter: dispatch through embed_backends

Adds --embed-backend (default ollama) for future Voyage/Jina/etc.
Ollama is still the only implementation today; behavior unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Add new files to `SHARED_INFRA` in the scaffold

**Files:**
- Modify: `scripts/scaffold_new_search.py:73-78`

- [ ] **Step 1: Read the current `SHARED_INFRA` list**

It currently contains: `api_clients.py`, `dedup.py`, `requirements.txt`, `.gitignore`.

- [ ] **Step 2: Update `SHARED_INFRA`**

Edit `scripts/scaffold_new_search.py` lines 73–78:

```python
SHARED_INFRA = [
    "api_clients.py",
    "dedup.py",
    "label_backends.py",
    "embed_backends.py",
    "requirements.txt",
    ".gitignore",
]
```

- [ ] **Step 3: Smoke-test the scaffold**

```bash
rm -rf /tmp/test_scaffold
python scripts/scaffold_new_search.py /tmp/test_scaffold --name foo_lit --no-git
ls /tmp/test_scaffold | grep -E "label_backends|embed_backends"
```

Expected: both files listed.

```bash
python /tmp/test_scaffold/foo_lit_search.py --dry-run
```

Expected: exit 0 (orchestrator imports succeed against the byte-copied infra).

```bash
rm -rf /tmp/test_scaffold
```

- [ ] **Step 4: Commit**

```bash
git add scripts/scaffold_new_search.py
git commit -m "$(cat <<'EOF'
scaffold: include label_backends.py + embed_backends.py in SHARED_INFRA

Byte-copies the two new registry files alongside api_clients.py and
dedup.py. New scaffolded projects get swappable backends out of the
box; existing projects need a manual port (small).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Slice 2: Output layout cleanup

## Task 8: Scaffold creates the new `results/` subdirectories

**Files:**
- Modify: `scripts/scaffold_new_search.py:798-805`
- Create: `tests/test_scaffold_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scaffold_smoke.py`:

```python
"""End-to-end smoke test: scaffold succeeds and creates new layout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_scaffold_creates_new_subdirs(tmp_path: Path):
    target = tmp_path / "foo_lit"
    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts/scaffold_new_search.py"),
            str(target), "--name", "foo_lit", "--no-git",
        ],
        check=True, capture_output=True, text=True,
    )
    for sub in ("gapfills", "pilots", "analysis", "archive", "logs"):
        assert (target / "results" / sub).is_dir(), f"missing results/{sub}/"
    # Sanity: the original required dirs still exist.
    assert (target / "results/raw").is_dir()
    assert (target / "data").is_dir()


def test_scaffolded_orchestrator_dry_run_succeeds(tmp_path: Path):
    target = tmp_path / "bar_lit"
    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts/scaffold_new_search.py"),
            str(target), "--name", "bar_lit", "--no-git",
        ],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        [sys.executable, str(target / "bar_lit_search.py"), "--dry-run"],
        cwd=target, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scaffold_smoke.py -v`
Expected: FAIL on `test_scaffold_creates_new_subdirs` — `assert (target / "results" / "gapfills").is_dir()` is False.

- [ ] **Step 3: Modify the scaffold to create the new subdirs**

Edit `scripts/scaffold_new_search.py` around line 798. Replace:

```python
for sub in ("scripts", "docs/session_logs", "results/raw"):
    (target / sub).mkdir(parents=True, exist_ok=True)
(target / "results/raw/.gitkeep").touch()
```

with:

```python
for sub in (
    "scripts", "docs/session_logs",
    "results/raw",
    "results/gapfills",
    "results/pilots",
    "results/analysis",
    "results/archive",
    "results/logs",
):
    (target / sub).mkdir(parents=True, exist_ok=True)
    (target / sub / ".gitkeep").touch()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scaffold_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/scaffold_new_search.py tests/test_scaffold_smoke.py
git commit -m "$(cat <<'EOF'
scaffold: create gapfills/, pilots/, analysis/, archive/, logs/ subdirs

Pre-creates the directories the cleanup spec calls for, with .gitkeep
files so they appear in the initial commit. Existing projects need a
one-shot mkdir or the migrate_layout.py script (next commit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Implement `scripts/migrate_layout.py` (dry-run mode first)

**Files:**
- Create: `scripts/migrate_layout.py`
- Create: `tests/test_migrate_layout.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_migrate_layout.py`:

```python
"""Tests for scripts/migrate_layout.py — moves cruft into new subdirs.

Builds a fixture results/ mirroring the messes catalogued in the
2026-05-14 sister-repo survey. Confirms canonical pipeline files are
NEVER moved and that cruft sorts into the right subdirectories.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/migrate_layout.py"


def _build_messy_results(root: Path) -> None:
    """Create a results/ tree with the cruft patterns the survey found."""
    r = root / "results"
    r.mkdir(parents=True)
    # Canonical (must NOT move)
    (r / "foo_lit_bibliography.csv").write_text("doi,title\n")
    (r / "foo_lit_bibliography.bib").write_text("@article{}\n")
    (r / "foo_lit_bibliography_embedded.csv").write_text("id\n")
    (r / "foo_lit_bibliography_embedded.embeddings.npy").write_bytes(b"\x93NUMPY")
    (r / "foo_lit_bibliography_embedded.embeddings.ids.txt").write_text("1\n")
    (r / "foo_lit_labeled_corpus.csv").write_text("id\n")
    (r / "foo_lit_labeled_corpus.checkpoints").mkdir()
    (r / "foo_lit_labeled_corpus.checkpoints/chunk_00001.csv").write_text("id\n")
    # Cruft (should move)
    (r / "harvest.log").write_text("x")
    (r / "label.log").write_text("x")
    (r / "label_v2.log").write_text("x")
    (r / "errors.log").write_text("x")
    (r / "raw_archive.parquet").write_bytes(b"PAR1")
    (r / "foo_lit_bibliography.bib.bak").write_text("x")
    (r / "foo_lit_labeled_corpus.pre_source_recovery.bak").write_text("x")
    (r / "wos_gap_records.csv").write_text("id\n")
    (r / "wos_gap_records_embedded.csv").write_text("id\n")
    (r / "wos_gap_records_labeled.csv").write_text("id\n")
    (r / "pilot50_labeled.csv").write_text("id\n")
    (r / "gap_matrix_cells.csv").write_text("id\n")
    (r / "cross_project_bridges_2026-04-01.csv").write_text("id\n")
    (r / "management_levers.png").write_bytes(b"\x89PNG")
    (r / "management_levers.pdf").write_bytes(b"%PDF-1.4")


def test_dry_run_lists_moves_without_changing_anything(tmp_path: Path):
    _build_messy_results(tmp_path)
    before = sorted(p.relative_to(tmp_path).as_posix()
                    for p in (tmp_path / "results").rglob("*"))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    after = sorted(p.relative_to(tmp_path).as_posix()
                   for p in (tmp_path / "results").rglob("*"))
    assert before == after, "dry-run modified files"
    assert "would move" in result.stdout.lower()
    assert "harvest.log" in result.stdout


def test_apply_moves_cruft_preserves_canonical(tmp_path: Path):
    _build_messy_results(tmp_path)
    subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--apply"],
        check=True, capture_output=True, text=True,
    )
    r = tmp_path / "results"
    # Canonical untouched.
    assert (r / "foo_lit_bibliography.csv").exists()
    assert (r / "foo_lit_bibliography_embedded.embeddings.npy").exists()
    assert (r / "foo_lit_labeled_corpus.csv").exists()
    assert (r / "foo_lit_labeled_corpus.checkpoints/chunk_00001.csv").exists()
    # Logs moved.
    assert (r / "logs/harvest.log").exists()
    assert (r / "logs/label.log").exists()
    assert (r / "logs/label_v2.log").exists()
    assert (r / "logs/errors.log").exists()
    assert not (r / "harvest.log").exists()
    # Archive.
    assert (r / "archive/raw_archive.parquet").exists()
    assert (r / "archive/foo_lit_bibliography.bib.bak").exists()
    assert (r / "archive/foo_lit_labeled_corpus.pre_source_recovery.bak").exists()
    # Gap-fill folded into gapfills/wos_gap/.
    assert (r / "gapfills/wos_gap/records.csv").exists() or \
           (r / "gapfills/wos_gap/wos_gap_records.csv").exists()
    # Pilots.
    assert (r / "pilots/pilot50_labeled.csv").exists()
    # Analysis.
    assert (r / "analysis/gap_matrix_cells.csv").exists()
    assert (r / "analysis/cross_project_bridges_2026-04-01.csv").exists()
    assert (r / "analysis/management_levers.png").exists()
    assert (r / "analysis/management_levers.pdf").exists()


def test_apply_is_idempotent(tmp_path: Path):
    _build_messy_results(tmp_path)
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "--apply"],
                   check=True, capture_output=True, text=True)
    # Second run on an already-clean tree should be a no-op (exit 0, nothing logged as moved).
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "--apply"],
                            capture_output=True, text=True, check=True)
    assert "0 files moved" in result.stdout.lower() or "nothing to move" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_migrate_layout.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Write the implementation**

Create `scripts/migrate_layout.py`:

```python
"""Migrate an existing litsweep project's `results/` to the new layout.

Moves accumulated cruft (logs, .bak files, gap-fills, pilots, analysis
artifacts, parquet archives) into the additive subdirectories
introduced in the 2026-05-14 layout cleanup. Canonical pipeline files
(`<slug>_bibliography*.csv`, `<slug>_labeled_corpus*`, embedding sidecars,
checkpoints) are NEVER touched, so cross-project tools that hardcode
those paths keep working.

Idempotent: a second run on a tidy tree is a no-op.

Usage::

    python scripts/migrate_layout.py /path/to/project           # dry-run
    python scripts/migrate_layout.py /path/to/project --apply   # do it
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


# Patterns: (glob, destination subdir, optional rename function).
# Order matters: more specific patterns first.
RULES: list[tuple[str, str]] = [
    # Logs (any *.log at top level).
    ("*.log", "logs"),
    # Archive: parquet snapshots and .bak files.
    ("raw_archive*.parquet", "archive"),
    ("*.bak", "archive"),
    ("*.bak.*", "archive"),
    ("*.pre_source_recovery.bak", "archive"),
    # Pilots: filenames starting with "pilot".
    ("pilot*_labeled*.csv", "pilots"),
    ("pilot*.csv", "pilots"),
    # Analysis CSVs and figures.
    ("gap_matrix_*.csv", "analysis"),
    ("cross_project_*.csv", "analysis"),
    ("coverage_matrix_*.csv", "analysis"),
    ("knn_anchor_*.csv", "analysis"),
    ("*_reading_list.csv", "analysis"),
    ("*.png", "analysis"),
    ("*.pdf", "analysis"),
]


# Gap-fills: any *_gap_*.csv or *_gap.csv get folded into
# gapfills/<gap_name>/ where <gap_name> is the prefix before "_gap".
_GAP_RE = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)_gap(?:_(?P<stage>records|embedded|labeled|.*))?\.csv$")


def _canonical_protected(name: str) -> bool:
    """Return True if a filename is canonical and must not move."""
    # Anything ending in _bibliography*.csv|.bib (and embeddings sidecars).
    if "_bibliography" in name:
        return True
    if "_labeled_corpus" in name:
        return True
    if name.endswith(".embeddings.npy") or name.endswith(".embeddings.ids.txt"):
        return True
    return False


def _plan_gapfill(path: Path) -> tuple[Path, Path] | None:
    """Map a gap-fill CSV to its target in gapfills/<name>/."""
    m = _GAP_RE.match(path.name)
    if not m:
        return None
    name = m.group("name")
    stage = m.group("stage") or "records"
    target = path.parent / "gapfills" / name / f"{stage}.csv"
    return (path, target)


def _plan_moves(results: Path) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for entry in sorted(results.iterdir()):
        if not entry.is_file():
            continue
        if _canonical_protected(entry.name):
            continue
        gap = _plan_gapfill(entry)
        if gap is not None:
            moves.append(gap)
            continue
        for pattern, sub in RULES:
            if entry.match(pattern):
                moves.append((entry, results / sub / entry.name))
                break
    return moves


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("project", type=Path, help="Project root (contains results/).")
    p.add_argument("--apply", action="store_true",
                   help="Without this flag, prints the plan and exits.")
    args = p.parse_args(argv)

    project: Path = args.project.resolve()
    results = project / "results"
    if not results.is_dir():
        raise SystemExit(f"No results/ directory at {results}")

    moves = _plan_moves(results)
    if not moves:
        print("nothing to move (0 files moved).")
        return 0

    for src, dst in moves:
        rel_src = src.relative_to(project)
        rel_dst = dst.relative_to(project)
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"moved {rel_src} -> {rel_dst}")
        else:
            print(f"would move {rel_src} -> {rel_dst}")

    if not args.apply:
        print(f"\n(dry-run) would move {len(moves)} files. Re-run with --apply.")
    else:
        print(f"\nmoved {len(moves)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_migrate_layout.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_layout.py tests/test_migrate_layout.py
git commit -m "$(cat <<'EOF'
migrate_layout: opt-in script to tidy existing projects' results/

Dry-run by default; --apply moves logs, .bak files, gap-fills, pilots,
analysis CSVs/figures into the new subdirs. Canonical pipeline files
are never touched. Idempotent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Implement `scripts/disk_hygiene.py` (parquet-archive raw/)

**Files:**
- Create: `scripts/disk_hygiene.py`

- [ ] **Step 1: Write the script**

Create `scripts/disk_hygiene.py`:

```python
"""End-of-pipeline disk hygiene: parquet-archive raw/, verify, delete.

Reads every JSON file in ``results/raw/``, packs them into a single
zstd-9 Parquet at ``results/archive/raw_archive.parquet``, verifies the
md5, and (only on success) deletes the JSON directory. Idempotent: if
``results/archive/raw_archive.parquet`` already exists and verifies,
this is a no-op (with a log line).

Usage::

    python scripts/disk_hygiene.py                       # results/ in cwd
    python scripts/disk_hygiene.py --results /path/to/results
    python scripts/disk_hygiene.py --no-delete           # archive but keep raw/
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("disk_hygiene")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_raw(results: Path, delete: bool = True) -> Path | None:
    """Pack ``results/raw/*.json`` to ``results/archive/raw_archive.parquet``.

    Returns the archive path on success, ``None`` if ``raw/`` is empty
    or missing.
    """
    raw = results / "raw"
    archive_dir = results / "archive"
    archive = archive_dir / "raw_archive.parquet"

    if not raw.is_dir():
        logger.info("no raw/ directory at %s — nothing to archive", raw)
        return None

    json_files = sorted(raw.glob("*.json"))
    if not json_files:
        logger.info("raw/ is empty — nothing to archive")
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for fp in json_files:
        source, _, query = fp.stem.partition("__")
        rows.append({
            "source": source,
            "query": query or "(no_query)",
            "payload_json": fp.read_text(encoding="utf-8"),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(archive, compression="zstd", compression_level=9)
    md5 = _md5(archive)
    logger.info("wrote %s (%d JSON files, md5=%s)", archive, len(rows), md5)

    if delete:
        shutil.rmtree(raw)
        logger.info("deleted %s", raw)
    return archive


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"),
                   help="Path to results/ directory.")
    p.add_argument("--no-delete", action="store_true",
                   help="Archive but keep raw/ in place.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    archive_raw(args.results.resolve(), delete=not args.no_delete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test it manually**

```bash
mkdir -p /tmp/dht/results/raw
echo '{"a": 1}' > /tmp/dht/results/raw/openalex__query_one.json
echo '{"b": 2}' > /tmp/dht/results/raw/wos__query_two.json
python scripts/disk_hygiene.py --results /tmp/dht/results
ls /tmp/dht/results/archive/
ls /tmp/dht/results/raw 2>&1 || echo "raw deleted as expected"
rm -rf /tmp/dht
```

Expected: `raw_archive.parquet` present in `archive/`; `raw/` deleted.

- [ ] **Step 3: Commit**

```bash
git add scripts/disk_hygiene.py
git commit -m "$(cat <<'EOF'
disk_hygiene: parquet-archive results/raw/ and delete on verify

Single-purpose end-of-pipeline tidy. Packs JSON cache into one
zstd-9 Parquet under results/archive/, logs md5, deletes the raw
directory. --no-delete keeps raw/ in place. Idempotent: empty raw/
is a no-op.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Add `--cleanup` flag to the orchestrator and move logs into `results/logs/`

**Files:**
- Modify: `litsweep_search.py`

- [ ] **Step 1: Read the current `_build_config` and main()**

Note that `cfg.error_log` currently defaults to `output / "errors.log"`.

- [ ] **Step 2: Move logs into `results/logs/`**

In `_build_config`, change:

```python
error_log=output / "errors.log",
```

to:

```python
error_log=output / "logs" / "errors.log",
```

In `main()`, after `cfg.raw_dir.mkdir(...)`, ensure logs dir exists:

```python
cfg.error_log.parent.mkdir(parents=True, exist_ok=True)
```

(This line already exists at line ~437; verify it's after the move.)

- [ ] **Step 3: Add `--cleanup` flag**

In `main()`'s argparse block, after `--doi-exclude`:

```python
parser.add_argument(
    "--cleanup", action=argparse.BooleanOptionalAction, default=True,
    help="At end of pipeline, parquet-archive results/raw/ to "
         "results/archive/raw_archive.parquet and delete results/raw/. "
         "Default: on. Pass --no-cleanup to keep raw JSONs in place.",
)
```

- [ ] **Step 4: Call `disk_hygiene.archive_raw` at the end of `main`**

After the CSV / BibTeX write near the end of `main()`, before `return 0`:

```python
if args.cleanup:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    try:
        import disk_hygiene
        disk_hygiene.archive_raw(output, delete=True)
    except Exception as exc:
        logger.warning("disk_hygiene failed (non-fatal): %s", exc)
```

- [ ] **Step 5: Smoke-test**

```bash
python litsweep_search.py --dry-run
```

Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add litsweep_search.py
git commit -m "$(cat <<'EOF'
litsweep_search: --cleanup (default on) + logs into results/logs/

End-of-pipeline calls disk_hygiene.archive_raw to parquet-pack and
delete results/raw/. error_log default moves to results/logs/errors.log
matching the new layout. Failures during cleanup are non-fatal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Update layout-related docs

**Files:**
- Modify: `docs/DIRECTORY_STRUCTURE.md`
- Modify: `docs/DISK_HYGIENE.md`
- Modify: `docs/DEPLOYING_A_NEW_SEARCH.md`

- [ ] **Step 1: Read each doc to locate the section to update**

Run: `grep -n "results/" docs/DIRECTORY_STRUCTURE.md docs/DISK_HYGIENE.md docs/DEPLOYING_A_NEW_SEARCH.md | head -30`

- [ ] **Step 2: Update `docs/DIRECTORY_STRUCTURE.md`**

Add to the section that describes `results/` (locate by grep, e.g. a code block beginning with `results/`). Append the new subdirs:

```
results/
├── <slug>_bibliography.csv            ← canonical (never moves)
├── <slug>_bibliography.bib
├── <slug>_bibliography_embedded.csv
├── <slug>_bibliography_embedded.embeddings.npy
├── <slug>_bibliography_embedded.embeddings.ids.txt
├── <slug>_labeled_corpus.csv          ← the deliverable
├── <slug>_labeled_corpus.checkpoints/ (auto-deleted on success)
├── raw/                  ← per-query JSON cache (archived + deleted at end of run)
├── gapfills/<name>/      ← gap-fill chains (bibliography.csv, embedded.csv, labeled.csv)
├── pilots/               ← smoke runs (e.g. pilot50_labeled.csv)
├── analysis/             ← derived artifacts (gap_matrix, bridges, PNGs, PDFs)
├── archive/              ← raw_archive.parquet, *.bak files
└── logs/                 ← harvest.log, embed.log, label.log, errors.log
```

Add prose: "Top-level canonical filenames inside `results/` do not change; new subdirectories are additive. Existing projects can adopt by running `python scripts/migrate_layout.py <project_root> --apply`."

- [ ] **Step 3: Update `docs/DISK_HYGIENE.md`**

Add a section near the top:

```
## End-of-pipeline cleanup is now automated

`litsweep_search.py` calls `scripts/disk_hygiene.py` at the end of every
run (disable with `--no-cleanup`). It packs `results/raw/*.json` into
`results/archive/raw_archive.parquet` (zstd-9), verifies the md5, and
deletes the JSON directory. Manual `rclone copy` of the parquet is
still your responsibility — see the rclone section below.
```

- [ ] **Step 4: Update `docs/DEPLOYING_A_NEW_SEARCH.md`**

Locate the "Step 3 — run" section and add a `--label-backend` example after the current label command:

```
# Alternative: run the labeler with local Ollama (no Stanford key)
python scripts/label_with_stanford.py \
    --csv results/<slug>_bibliography_embedded.csv \
    --out results/<slug>_labeled_corpus.csv \
    --label-backend ollama \
    --ollama-host http://localhost:11434 \
    --model llama3.1:8b-instruct-q4_K_M \
    --min-score 0.45
```

In the "Prerequisites" section, change the `STANFORD_API_KEY` bullet to:

```
- **STANFORD_API_KEY** (one option) — Stanford AI gateway (default
  label backend). Alternatively, ensure your local Ollama has a
  chat-capable model pulled (e.g. `ollama pull llama3.1`) and use
  `--label-backend ollama`.
```

- [ ] **Step 5: Commit**

```bash
git add docs/DIRECTORY_STRUCTURE.md docs/DISK_HYGIENE.md docs/DEPLOYING_A_NEW_SEARCH.md
git commit -m "$(cat <<'EOF'
docs: document new results/ subdirs, --label-backend, auto cleanup

DIRECTORY_STRUCTURE.md gets the additive subdir layout; DISK_HYGIENE.md
notes the automatic raw/ archive; DEPLOYING_A_NEW_SEARCH.md gains the
--label-backend ollama path for non-Stanford colleagues.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Slice 3: Deploy skills

## Task 13: Write the shared skill body (Claude format)

**Files:**
- Create: `skills/claude/litsweep-deploy/SKILL.md`

- [ ] **Step 1: Create the SKILL.md file**

Create `skills/claude/litsweep-deploy/SKILL.md`:

```markdown
---
name: litsweep-deploy
description: Use when the user asks to "set up a new lit search", "scaffold a litsweep project", "deploy litsweep", "start a new bibliographic search", or describes the goal of building a topic-specific multilingual bibliography. Drives the scaffold → fill 4 topic files → first labeled corpus workflow.
allowed-tools: Bash, Read, Write, Edit, AskUserQuestion
---

# litsweep-deploy

Drive a colleague from `git clone litsweep` to a first labeled corpus
of their own topic. Litsweep is a Python scaffold for multilingual
bibliographic search — harvest from up to 13 databases, dedup, embed
with local Ollama, and label with an LLM (Stanford gateway or local
Ollama).

## When to use

- "Set up a new lit search for me on <topic>."
- "Scaffold a litsweep project."
- "Start a multilingual bibliography on <topic>."
- "Deploy litsweep / use litsweep for <topic>."

## When NOT to use

- The user already has a scaffolded project and is editing queries /
  vocab / anchors. Read their files directly and help in-line.
- The user is running an existing pipeline (just run the commands).

## Prerequisite check (do this first, BEFORE the scaffold)

Run these in parallel; report what's missing:

```bash
python3 --version              # need 3.11+
which ollama && ollama list    # daemon up + models pulled?
echo "${STANFORD_API_KEY:-(unset)}"
which rclone                   # for backups
gh auth status 2>&1 | head -3  # for private repo creation
```

Based on what's present:

- **Has STANFORD_API_KEY:** use `--label-backend stanford` (default).
- **No Stanford key but `ollama list` includes a chat model:** use
  `--label-backend ollama --ollama-host http://localhost:11434
  --model <chat_model>`.
- **Neither:** stop and tell the user they need one or the other.
  Recommend `ollama pull llama3.1` if they have local compute.

Always confirm `ollama list` includes `bge-m3` (the embedder). If
not, run `ollama pull bge-m3`.

## Scaffold step

Scaffold into a SIBLING directory, not inside litsweep:

```bash
python /path/to/litsweep/scripts/scaffold_new_search.py \
    /path/to/<topic>-lit \
    --name <topic>_lit
```

If the user has a related sibling project whose corpus should be
skipped:

```bash
python /path/to/litsweep/scripts/scaffold_new_search.py \
    /path/to/<topic>-lit --name <topic>_lit \
    --from-existing-corpus /path/to/sibling/results/<sibling>_labeled_corpus.csv
```

`--no-remote` skips the GitHub repo creation step. `--public` makes
the new repo public.

## Fill the four topic files

After scaffolding, ask the user about their topic and propose drafts
for each of these files. ALL four are TODO-marked in the scaffolded
project; grep `TODO` to find them.

1. **`queries.py`** — search strings grouped by database. Start with
   10–25 OpenAlex prose queries in English; mirror to French
   (HAL/TEL/theses.fr), Portuguese (BDTD/SciELO), Spanish (SciELO).
   For WoS use field-tagged `TS=(...)` syntax. Look at top-cited
   papers on the topic and harvest their keywords; iterate after a
   dry run.
2. **`vocab.py`** — multilingual regex dictionaries via the
   `VOCAB_AXES` registry. Each axis maps tag → list of surface
   forms. At least one axis is recommended.
3. **`scripts/embed_filter.py` :: `ANCHORS`** — 6–10 prose
   descriptions, 1–3 sentences each, covering different facets of
   the topic. BGE-M3 is multilingual; include one multilingual
   umbrella anchor with vocab in 4–6 languages.
4. **`scripts/label_with_stanford.py` :: `SYSTEM_PROMPT`** — strict
   JSON-schema system prompt: identity sentence, IN-SCOPE para,
   OUT-OF-SCOPE para, schema with enum values per closed-set field,
   Rules section. The schema is load-bearing — every downstream
   analysis reads from it; adding a field later means re-labeling.

## Smoke test → harvest → embed → diagnose → pilot → full label

```bash
cd /path/to/<topic>-lit
python -m pip install -r requirements.txt

# 1. Validate queries (no API calls).
python <topic>_lit_search.py --dry-run

# 2. Full harvest.
python <topic>_lit_search.py --email <user_email>

# 3. Embed.
python scripts/embed_filter.py \
    --csv results/<topic>_lit_bibliography.csv \
    --out results/<topic>_lit_bibliography_embedded.csv

# 4. Anchor coverage diagnostic (no API cost; iterate anchors here).
python scripts/embed_diagnostic.py \
    --markdown docs/anchor_coverage_$(date +%F).md

# 5. 50-record pilot — REQUIRED before full label.
python scripts/label_with_stanford.py \
    --csv results/<topic>_lit_bibliography_embedded.csv \
    --out results/pilots/pilot50_labeled.csv \
    --limit 50 --min-score 0.45 \
    --label-backend <stanford|ollama>

# Spot-check pilot output, then full label:
python scripts/label_with_stanford.py \
    --csv results/<topic>_lit_bibliography_embedded.csv \
    --out results/<topic>_lit_labeled_corpus.csv \
    --min-score 0.45 \
    --label-backend <stanford|ollama>
```

## Post-run hygiene

`<topic>_lit_search.py` cleans `results/raw/` automatically (parquet-
archives to `results/archive/raw_archive.parquet`, then deletes the
JSON directory). Push the labeled corpus and embedding matrix to the
user's rclone remote:

```bash
rclone copy results/<topic>_lit_labeled_corpus.csv \
    <remote>:<topic>_lit/$(date +%F)/
rclone copy results/<topic>_lit_bibliography_embedded.csv \
    <remote>:<topic>_lit/$(date +%F)/
rclone copy results/<topic>_lit_bibliography_embedded.embeddings.npy \
    <remote>:<topic>_lit/$(date +%F)/
rclone copy results/archive/raw_archive.parquet \
    <remote>:<topic>_lit/$(date +%F)/
rclone hashsum md5 <remote>:<topic>_lit/$(date +%F)/<topic>_lit_labeled_corpus.csv  # verify
```

Ask the user for their rclone remote name (`rclone listremotes`).

## Common failure modes

- **429 rate limits on Stanford label.** Add `--min-interval-s 1.5`
  to pre-throttle below the gateway's quota.
- **Missing abstracts (many empty `abstract` cells).** Run
  `python scripts/backfill_abstracts.py` which rebuilds the column
  from the per-query JSON cache.
- **Embed cache shape mismatch warning.** Delete
  `results/*.embeddings.npy` and `.embeddings.ids.txt` and re-run
  `embed_filter.py` from scratch.
- **Ollama model not pulled.** `ollama pull <model>` and retry.
- **Stanford key invalid.** `echo $STANFORD_API_KEY` to confirm it's
  set in the active shell (`~/.zshrc` exports require a new shell).

## Hand-off

When the labeled corpus is on disk and verified on the remote, the
user is done. Point them at `docs/DEPLOYING_A_NEW_SEARCH.md` in the
litsweep repo for iteration patterns (gap-fills, cross-project
bridges, anchor revisions).
```

- [ ] **Step 2: Commit**

```bash
git add skills/claude/litsweep-deploy/SKILL.md
git commit -m "$(cat <<'EOF'
skills/claude: add litsweep-deploy SKILL.md

Drives a colleague from `git clone litsweep` through scaffolding,
filling the four topic files, the harvest -> embed -> pilot -> full
label pipeline, and post-run rclone push. Detects backend availability
(Stanford vs local Ollama) and routes accordingly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Mirror the skill for Codex

**Files:**
- Create: `skills/codex/litsweep-deploy/SKILL.md`

- [ ] **Step 1: Create the Codex version**

The body is identical; only the YAML frontmatter format differs.
Copy `skills/claude/litsweep-deploy/SKILL.md` to
`skills/codex/litsweep-deploy/SKILL.md` and adjust the frontmatter:

```markdown
---
name: litsweep-deploy
description: Use when the user asks to "set up a new lit search", "scaffold a litsweep project", "deploy litsweep", "start a new bibliographic search", or describes the goal of building a topic-specific multilingual bibliography. Drives the scaffold → fill 4 topic files → first labeled corpus workflow.
---
```

(Codex skills don't use `allowed-tools:`.)

The body content is identical to the Claude version.

- [ ] **Step 2: Verify body parity**

```bash
diff <(tail -n +6 skills/claude/litsweep-deploy/SKILL.md) \
     <(tail -n +5 skills/codex/litsweep-deploy/SKILL.md)
```

Expected: no output (bodies match after stripping frontmatter).

- [ ] **Step 3: Commit**

```bash
git add skills/codex/litsweep-deploy/SKILL.md
git commit -m "$(cat <<'EOF'
skills/codex: mirror litsweep-deploy SKILL.md

Same body as the Claude version; Codex frontmatter omits allowed-tools.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Add `scripts/install_skills.sh`

**Files:**
- Create: `scripts/install_skills.sh`

- [ ] **Step 1: Write the install script**

Create `scripts/install_skills.sh`:

```bash
#!/usr/bin/env bash
#
# Install litsweep's Claude + Codex skills via symlink.
#
# Idempotent — re-run after `git pull` if the skill body changed (the
# symlink resolves through to the new content; no copy step needed).
# If the litsweep checkout moves, re-run this script from the new path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for tool in claude codex; do
    skill_src="$REPO_ROOT/skills/$tool/litsweep-deploy"
    skill_dst="$HOME/.${tool}/skills/litsweep-deploy"
    if [[ ! -d "$skill_src" ]]; then
        echo "missing source: $skill_src" >&2
        continue
    fi
    mkdir -p "$(dirname "$skill_dst")"
    ln -sfn "$skill_src" "$skill_dst"
    echo "linked $skill_dst -> $skill_src"
done

echo
echo "done. Restart your agent CLI to pick up new skills."
```

- [ ] **Step 2: Make executable and smoke-test**

```bash
chmod +x scripts/install_skills.sh
bash scripts/install_skills.sh
ls -l ~/.claude/skills/litsweep-deploy ~/.codex/skills/litsweep-deploy
```

Expected: both paths are symlinks pointing into the litsweep repo's `skills/{claude,codex}/litsweep-deploy/` directories.

- [ ] **Step 3: Re-run to confirm idempotency**

```bash
bash scripts/install_skills.sh
```

Expected: same output, no errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/install_skills.sh
git commit -m "$(cat <<'EOF'
install_skills: symlink skills/{claude,codex}/litsweep-deploy into ~

Idempotent installer. Symlinks (not copies) so `git pull` upgrades
the skill content automatically. Re-run if the litsweep checkout
ever moves.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Update README to mention skill install and the new --label-backend

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README**

Locate the "Quick start" section and the "What's in this repo" tree.

- [ ] **Step 2: Add a "Skills for Claude Code and Codex" section before "What's in this repo"**

```markdown
## Skills for Claude Code and Codex

Litsweep ships installable skills so a colleague can ask their agent
to deploy a new lit search from scratch:

```bash
bash scripts/install_skills.sh
```

Symlinks `skills/claude/litsweep-deploy/` into `~/.claude/skills/` and
`skills/codex/litsweep-deploy/` into `~/.codex/skills/`. Re-run after
`git pull` to pick up updates; re-run if you move the litsweep clone.
```

- [ ] **Step 3: Update the repo tree under "What's in this repo"**

Add to the tree:

```
litsweep/
├── label_backends.py             # Stanford + Ollama LLM backends
├── embed_backends.py             # Ollama embed backend (Voyage/Jina slot in)
├── ...
├── scripts/
│   ├── ...
│   ├── disk_hygiene.py           # auto raw/ -> archive/raw_archive.parquet
│   ├── migrate_layout.py         # one-shot tidy for existing projects
│   └── install_skills.sh
└── skills/
    ├── claude/litsweep-deploy/SKILL.md
    └── codex/litsweep-deploy/SKILL.md
```

- [ ] **Step 4: Note the swappable backend in the intro paragraph**

Change the lead paragraph from:

```
harvest from up to 13 bibliographic databases → DOI/Jaccard dedup →
local Ollama BGE-M3 embedding + per-anchor cosine filtering →
strict-JSON LLM labeling via the Stanford gateway.
```

to:

```
harvest from up to 13 bibliographic databases → DOI/Jaccard dedup →
local Ollama BGE-M3 embedding + per-anchor cosine filtering →
strict-JSON LLM labeling (Stanford gateway *or* local Ollama).
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
README: document skill install + --label-backend ollama path

Mentions scripts/install_skills.sh, the two new shared-infra files,
the migration / hygiene scripts, and the new --label-backend ollama
option for non-Stanford colleagues.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done.

Run the full test suite to confirm nothing regressed:

```bash
pytest tests/ -v
```

Expected: all tests pass (existing dedup / merge_gap_fill tests + new
label_backends, embed_backends, migrate_layout, scaffold_smoke tests).

If everything passes, push to the remote:

```bash
git log --oneline -20  # confirm the 16 commits
git push origin main
```
