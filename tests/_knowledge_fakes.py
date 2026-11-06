"""In-memory doubles for the corpus, so knowledge tests need no Postgres.

Named with a leading underscore so pytest does not collect it as a test module.
The fake cursor dispatches on the leading SQL keyword, which is enough to
exercise the ingestion and search code paths and keeps the assertions about
*which* statements ran, not about a real database's behaviour.
"""

from __future__ import annotations


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self._result = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._conn.statements.append((" ".join(str(sql).split()), params))
        self._result = self._conn.respond(sql, params)
        self.rowcount = self._conn.rowcount
        return self

    def executemany(self, sql, seq):
        rows = list(seq)
        self._conn.statements.append((" ".join(str(sql).split()), rows))
        self._conn.many.append((" ".join(str(sql).split()), rows))
        self._result = None
        return self

    def fetchone(self):
        if isinstance(self._result, list):
            return self._result[0] if self._result else None
        return self._result

    def fetchall(self):
        if self._result is None:
            return []
        return self._result if isinstance(self._result, list) else [self._result]


class FakeConn:
    """Records statements and replays canned responses.

    ``responses`` maps an uppercase SQL keyword-prefix to either a value or a
    callable ``(sql, params) -> value``. Anything unmatched returns None.
    """

    def __init__(self, responses=None, rowcount=0):
        self.statements: list[tuple[str, object]] = []
        self.many: list[tuple[str, list]] = []
        self.responses = responses or {}
        self.rowcount = rowcount
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, *args, **kwargs):
        return FakeCursor(self)

    def respond(self, sql, params):
        text = " ".join(str(sql).split()).upper()
        for prefix, value in self.responses.items():
            if text.startswith(prefix.upper()):
                return value(sql, params) if callable(value) else value
        return None

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    # --- assertions helpers ---
    def sql_matching(self, needle: str) -> list[str]:
        return [s for s, _ in self.statements if needle.upper() in s.upper()]

    def ran(self, needle: str) -> bool:
        return bool(self.sql_matching(needle))


class FakeEmbedder:
    """Deterministic embedder: no model, no network, stable vectors."""

    def __init__(self, dim: int = 4, fail: bool = False):
        self.dim = dim
        self.fail = fail
        self.name = "fake"
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            return []
        return [[float(len(t) % 7) / 7.0] * self.dim for t in texts]

    def encode_one(self, text):
        vectors = self.encode([text])
        return vectors[0] if vectors else None
