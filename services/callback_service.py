import logging
import time

import requests

logger = logging.getLogger(__name__)


class CallbackService:
    def __init__(self, timeout: int = 30, max_attempts: int = 3, backoff: float = 1):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff = backoff

    def send(self, callback_url: str, payload: dict) -> None:
        for attempt in range(self.max_attempts):
            try:
                # Only a 2xx response acknowledges delivery; do not redirect job results.
                with requests.post(callback_url, json=payload, timeout=self.timeout,
                                   allow_redirects=False) as response:
                    if not 200 <= response.status_code < 300:
                        raise requests.HTTPError(f"Callback HTTP {response.status_code}", response=response)
                return
            except requests.RequestException as error:
                retryable = (error.response is None or error.response.status_code in (408, 429)
                             or error.response.status_code >= 500)
                if not retryable or attempt + 1 == self.max_attempts:
                    raise RuntimeError("Callback delivery failed") from error
                logger.warning("Callback attempt %s failed; retrying", attempt + 1)
                time.sleep(self.backoff * 2 ** attempt)
