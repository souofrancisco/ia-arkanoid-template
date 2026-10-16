"""Tick-indexed paddle tracker: infers the action delay from the agent's own moves and dead-reckons the paddle.

Ticks come from the server clock (round(time_elapsed * 30)), so skipped states (a slow agent) are handled: a tick
without a decision counts as "no move".
"""

STEP = 28.0
MAX_LAT = 12
KEEP = 48  # ticks of history


class PaddleTracker:
    def __init__(self, paddle_width: float, field_width: float) -> None:
        self.hi = field_width - paddle_width
        self.obs: dict[int, float] = {}
        self.sent: dict[int, int] = {}  # tick -> -1 / +1 (absent = no move)
        self.lat = 0

    def _clamp(self, x: float) -> float:
        return min(self.hi, max(0.0, x))

    def _mismatches(self, lat: int) -> tuple[int, int]:
        """Replay our own moves between consecutive observations (any gap) under a latency hypothesis."""
        bad = n = 0
        ticks = sorted(self.obs)
        for t1, t2 in zip(ticks, ticks[1:], strict=False):
            if not any(self.sent.get(s - k) for s in range(t1, t2) for k in range(MAX_LAT + 1)):
                continue  # no move could explain a change: uninformative
            x = self.obs[t1]
            for s in range(t1, t2):
                x = self._clamp(x + self.sent.get(s - lat, 0) * STEP)
            n += 1
            bad += abs(x - self.obs[t2]) > 0.5
        return bad, n

    def observe(self, tick: int, px: float) -> tuple[float, int]:
        """Record the paddle x seen at `tick`; returns (effective x including moves still in flight, latency)."""
        self.obs[tick] = px
        for d in (self.obs, self.sent):
            for t in [t for t in d if t < tick - KEEP]:
                del d[t]
        scores = [self._mismatches(lat) for lat in range(MAX_LAT + 1)]
        if scores[0][1] >= 4:
            best = min(s[0] for s in scores)
            if scores[self.lat][0] > best:  # keep the current estimate unless another is strictly better
                self.lat = [s[0] for s in scores].index(best)
        eff = px
        for t in range(tick - self.lat, tick):
            eff = self._clamp(eff + self.sent.get(t, 0) * STEP)
        return eff, self.lat

    def record(self, tick: int, move: int) -> None:
        if move:
            self.sent[tick] = move
