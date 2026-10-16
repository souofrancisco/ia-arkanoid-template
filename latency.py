"""Infer the action delay from the agent's own history and dead-reckon where the paddle will be."""

from collections import deque

STEP = 28.0
MAX_LAT = 3
WINDOW = 14


class PaddleTracker:
    """Per tick call update(observed paddle x, move sent at the previous tick); returns (effective x, latency).

    Tick t shows px_t. Decision d_t is applied before update t+lat, so px_{t+1} - px_t is the effect of d_{t-lat} and the
    moves not yet visible in px_t are the last `lat` decisions.
    """

    def __init__(self, paddle_width: float, field_width: float) -> None:
        self.hi = field_width - paddle_width
        self.obs: deque[float] = deque(maxlen=WINDOW)
        self.sent: deque[int] = deque(maxlen=WINDOW - 1)  # sent[i] is the decision taken right after obs[i]
        self.lat = 0

    def _clamp(self, x: float) -> float:
        return min(self.hi, max(0.0, x))

    def _mismatches(self, lat: int) -> int:
        bad = 0
        for i in range(lat, len(self.obs) - 1):
            expect = self._clamp(self.obs[i] + self.sent[i - lat] * STEP) - self.obs[i]
            bad += abs((self.obs[i + 1] - self.obs[i]) - expect) > 0.5
        return bad

    def update(self, px: float, last_sent: int) -> tuple[float, int]:
        if self.obs:
            self.sent.append(last_sent)
        self.obs.append(px)
        if len(self.obs) >= 6:
            scores = [self._mismatches(lat) for lat in range(MAX_LAT + 1)]
            best = min(scores)
            if scores[self.lat] > best:  # keep the current estimate unless another is strictly better
                self.lat = scores.index(best)
        eff = px
        for m in list(self.sent)[len(self.sent) - self.lat:] if self.lat else []:
            eff = self._clamp(eff + m * STEP)
        return eff, self.lat
