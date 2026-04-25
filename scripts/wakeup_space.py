"""scripts/wakeup_space.py - 4-hour HF-Spaces wake-up pinger.

The plan's ``"safety check you do every 4 hours"`` step. Hits ``/healthz``
on the deployed Space; if Spaces has hibernated the container, the request
itself wakes it up. Emits exit code 0 on success, 1 on failure (so cron
can email you).

Usage::

    python -m scripts.wakeup_space --url https://your-username-qubit-medic.hf.space
    # or set the URL via env var:
    QUBIT_MEDIC_URL=https://... python -m scripts.wakeup_space

To run forever from a laptop::

    while true; do
        python -m scripts.wakeup_space --url ... || echo "WAKE FAILED" | tee -a wake.log
        sleep 14400  # 4 hours
    done
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable


def _ping(url: str, *, timeout: float, retries: int) -> int:
    import httpx
    target = url.rstrip("/") + "/healthz"
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        t0 = time.monotonic()
        try:
            r = httpx.get(target, timeout=timeout)
            dt = time.monotonic() - t0
            if r.status_code == 200:
                print(f"[{time.strftime('%H:%M:%S')}] OK  {target} "
                      f"({dt * 1000:.0f}ms)  -> {r.json()}")
                return 0
            print(f"[{time.strftime('%H:%M:%S')}] HTTP {r.status_code} "
                  f"on attempt {attempt}/{retries} ({dt * 1000:.0f}ms)")
            # Spaces returns 503 while waking; back off and retry.
            time.sleep(min(30, 2 ** attempt))
        except Exception as exc:  # pragma: no cover - network-dependent
            last_exc = exc
            print(f"[{time.strftime('%H:%M:%S')}] error on attempt "
                  f"{attempt}/{retries}: {exc}")
            time.sleep(min(30, 2 ** attempt))
    print(f"[{time.strftime('%H:%M:%S')}] FAIL after {retries} attempts. "
          f"Last error: {last_exc}", file=sys.stderr)
    return 1


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("QUBIT_MEDIC_URL"),
                        help="Base URL of the Space (without /healthz).")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args(list(argv))

    if not args.url:
        print("ERROR: --url or $QUBIT_MEDIC_URL is required", file=sys.stderr)
        return 2
    return _ping(args.url, timeout=args.timeout, retries=args.retries)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
