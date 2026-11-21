"""Offline guarantee of the jury protocol, reproducible without Docker.

Under a hard socket guard (any IPv4/IPv6 socket, connection or name resolution raises) the real
protocol runs end to end through the public ``judge_adapter`` interface: generate_episode ->
make_env -> reset -> the rule-based baseline until the episode ends -> score_episode. The guard
has a positive control (it demonstrably blocks outgoing attempts) and records every attempt, so
a network call swallowed by the code under test would still fail the test. Scope: the guard
patches the ``socket`` module surface every stdlib/HTTP client uses (``socket.socket`` for
IPv4/IPv6, ``create_connection`` and the resolvers); a direct ``_socket.socket`` call or other
address families are out of scope. Two companion checks keep the shipped runtime lean:
``requirements.txt`` and the adapter's import graph carry no chart, numeric or HTTP libraries.
"""

from __future__ import annotations

import importlib.util
import json
import re
import socket
import subprocess
import sys
import tomllib
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

from fintrace.constants import MAX_STEPS
from fintrace.evaluator.runner import load_split_config
from fintrace.reward.official import official_reward

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_SEED = 10000
SCORE_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "return")
BLOCKED_FAMILIES = (-1, socket.AF_INET, socket.AF_INET6)  # -1 = CPython's default = AF_INET
RESOLVERS = ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex", "getnameinfo")
HEAVY_OR_NETWORK = (
    "torch", "matplotlib", "pandas", "numpy", "requests", "httpx", "urllib3", "pm4py",
    "streamlit", "transformers", "networkx",
)  # fmt: skip
RUNTIME_PINS = (
    "annotated-types", "pydantic", "pydantic-core", "pyyaml", "typing-extensions",
    "typing-inspection",
)  # fmt: skip


class NetworkBlocked(RuntimeError):
    """Raised by the guard. Deliberately not an OSError: http/urllib would wrap that into URLError
    and a 'try the network, fall back' path in the code under test could swallow it silently."""


Attempt = tuple[str, object]


@pytest.fixture
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Attempt]]:
    """Block IPv4/IPv6 sockets and name resolution for one test; yields the recorded attempts."""
    attempts: list[Attempt] = []
    real_socket = socket.socket

    class GuardedSocket(real_socket):  # type: ignore[misc,valid-type]
        def __init__(
            self, family: int = -1, type: int = -1, proto: int = -1, fileno: object = None
        ) -> None:
            if fileno is None and family in BLOCKED_FAMILIES:
                attempts.append(("socket", family))
                raise NetworkBlocked(f"socket(family={family}) blocked by the offline guard")
            super().__init__(family, type, proto, fileno)  # AF_UNIX / wrapped fds stay allowed

    def blocked(name: str) -> Callable[..., object]:
        def _blocked(*args: object, **kwargs: object) -> object:
            attempts.append((name, args[0] if args else kwargs))
            raise NetworkBlocked(f"socket.{name} blocked by the offline guard")

        return _blocked

    monkeypatch.setattr(socket, "socket", GuardedSocket)
    for name in RESOLVERS:
        monkeypatch.setattr(socket, name, blocked(name))
    yield attempts
    # monkeypatch undoes every patch after the test: the guard never leaks into other tests


def _load_driver() -> ModuleType:
    path = REPO_ROOT / "scripts" / "jury_protocol_smoke.py"
    spec = importlib.util.spec_from_file_location("jury_protocol_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _positive_control(attempts: list[Attempt]) -> None:
    """The guard demonstrably blocks every outgoing path a client library could take."""
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 80), timeout=0.01)
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(NetworkBlocked):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(NetworkBlocked):
        socket.socket()  # the default family resolves to AF_INET in CPython
    with pytest.raises(NetworkBlocked):
        urllib.request.urlopen("http://example.com", timeout=0.01)  # positive control
    assert [name for name, _ in attempts] == [
        "create_connection", "getaddrinfo", "socket", "socket", "create_connection",
    ]  # fmt: skip
    attempts.clear()


def test_jury_protocol_runs_offline(tmp_path: Path, _no_network: list[Attempt]) -> None:
    _positive_control(_no_network)
    driver = _load_driver()
    team = json.loads(load_split_config("dev").model_dump_json())
    summary = driver.run_protocol(DEV_SEED, tmp_path, team)
    assert _no_network == [], f"the protocol tried to use the network: {_no_network}"
    assert driver.check_summary(summary) == []
    assert summary["seed"] == DEV_SEED and summary["terminated"] and not summary["truncated"]
    assert 1 <= summary["steps"] <= MAX_STEPS
    assert summary["policy_rewards_zero"] and summary["info_empty"]  # closed mode (C19)
    assert summary["n_findings"] > 0
    score = summary["score"]
    assert tuple(score) == SCORE_KEYS
    tp, fp, fn = (score[k] for k in ("tp", "fp", "fn"))
    assert all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (tp, fp, fn))
    precision, recall, f1 = (float(score[k]) for k in ("precision", "recall", "f1"))
    assert all(0.0 <= v <= 1.0 for v in (precision, recall, f1))
    assert precision == pytest.approx(tp / (tp + fp)) if tp + fp else precision == 0.0
    assert recall == pytest.approx(tp / (tp + fn)) if tp + fn else recall == 0.0
    expected_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    assert f1 == pytest.approx(expected_f1)
    assert -1.0 <= float(score["return"]) <= 1.0
    assert float(score["return"]) == official_reward(f1, summary["steps"] - 1)
    assert float(score["return"]) > 0.5  # a real investigation, not an empty submission
    assert (tmp_path / "episode" / "manifest.json").is_file()


def test_runtime_requirements_lean() -> None:
    """The shipped image installs requirements.txt only: no chart, numeric or HTTP libraries."""
    pins = [
        line.split("==")[0].strip().lower()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert pins and all(re.fullmatch(r"[a-z0-9_.-]+", name) for name in pins)
    assert sorted(pins) == sorted(RUNTIME_PINS), pins  # exactly the six packages the image needs
    assert not set(pins) & set(HEAVY_OR_NETWORK), sorted(set(pins) & set(HEAVY_OR_NETWORK))
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    runtime = [re.split(r"[<>=!~ \[]", dep)[0].lower() for dep in project["dependencies"]]
    assert not set(runtime) & set(HEAVY_OR_NETWORK), runtime
    dev_deps = project["optional-dependencies"]["dev"]
    dev = [re.split(r"[<>=!~ \[]", dep)[0].lower() for dep in dev_deps]
    assert "matplotlib" in dev  # the charts still work from `make install`


def test_judge_adapter_imports_no_network_or_heavy_modules() -> None:
    """A fresh interpreter importing judge_adapter loads none of the heavy or HTTP packages."""
    probe = (
        "import sys, judge_adapter; "
        f"heavy = {set(HEAVY_OR_NETWORK)!r}; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))"
    )
    env = {"PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}", "PATH": "/usr/bin:/bin"}
    result = subprocess.run(  # noqa: S603  # fixed interpreter and argv, no shell
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
    )
    assert result.stdout.strip() == "[]", result.stdout

