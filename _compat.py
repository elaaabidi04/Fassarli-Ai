"""
Monkey-patch for Python 3.14 + urllib3 incompatibility.

Python 3.14's http.client.putheader() now strictly enforces Latin-1 encoding
for header values. urllib3 hasn't been updated to handle this yet, so any header
value containing non-Latin-1 characters (e.g. the em dash in the NVIDIA client's
User-Agent) raises UnicodeEncodeError. This patch replaces unencodable chars with
'?' before they reach http.client.
"""
import urllib3.connection as _u3conn

_orig_putheader = _u3conn.HTTPConnection.putheader


def _safe_putheader(self, header, *values):
    sanitized = []
    for v in values:
        if isinstance(v, str):
            v = v.encode("latin-1", errors="replace").decode("latin-1")
        sanitized.append(v)
    return _orig_putheader(self, header, *sanitized)


_u3conn.HTTPConnection.putheader = _safe_putheader
