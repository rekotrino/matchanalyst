import time
import requests

from ..config import SETTINGS


def get_json(url, params=None, retries=3, backoff=1.5):
    """GET with small retry loop. Returns parsed JSON or raises."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=SETTINGS.request_timeout,
                             headers={"User-Agent": SETTINGS.user_agent, "Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code} for {r.url}")
            if r.status_code in (404, 400):
                raise last
        except (requests.RequestException, ValueError) as e:  # noqa: PERF203
            last = e
        time.sleep(backoff * (attempt + 1))
    raise last
