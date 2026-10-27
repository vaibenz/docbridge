"""Who is asking, and which groups they are in.

Production: OpenID Connect. The web app signs the user in with any OIDC provider
(Google, Microsoft Entra ID, Okta, Auth0) and sends the ID or access token as
`Authorization: Bearer <token>`. The server verifies it locally:

* the signature, with the provider's public keys (JWKS, fetched and cached);
* only asymmetric algorithms (RS256, ES256 and friends). Accepting HS256 here
  would let an attacker sign tokens with the public key, and "none" would
  accept unsigned ones, so both are refused before the key is even looked up;
* issuer, audience, expiry and not-before, with a little clock skew;
* a subject, and for email based mapping a verified email.

Groups come from a claim in the token (`groups` by default, as Entra ID and Okta
can issue) and from an access file mapping emails and email domains to groups,
for providers like Google whose tokens carry no groups. Everyone also gets
`all`. Unknown users get only `all`, never more.

Development: DOCBRIDGE_AUTH=dev trusts an X-Dev-User header and X-Dev-Groups.
The server logs a warning on every request in that mode.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import jwt

ASYMMETRIC = ["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA"]


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    subject: str
    email: str | None
    groups: tuple[str, ...]


@dataclass
class AccessMap:
    """Emails and email domains to groups, from a JSON file:

        {"users": {"alice@harbourline.example": ["hr"]},
         "domains": {"harbourline.example": ["staff"]},
         "admins": ["ops@harbourline.example"]}
    """

    users: dict[str, list[str]] = field(default_factory=dict)
    domains: dict[str, list[str]] = field(default_factory=dict)
    admins: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None) -> "AccessMap":
        if not path:
            return cls()
        data = json.loads(Path(path).read_text())
        return cls({k.lower(): v for k, v in data.get("users", {}).items()},
                   {k.lower(): v for k, v in data.get("domains", {}).items()},
                   [a.lower() for a in data.get("admins", [])])

    def groups_for(self, email: str | None) -> set[str]:
        if not email:
            return set()
        email = email.lower()
        out = set(self.users.get(email, []))
        out |= set(self.domains.get(email.rsplit("@", 1)[-1], []))
        if email in self.admins:
            out.add("admin")
        return out


class TokenVerifier:
    def __init__(self, issuer: str, audience: str, key_for_token: Callable[[str], object],
                 access: AccessMap | None = None, groups_claim: str = "groups", leeway: int = 30):
        self.issuer, self.audience = issuer, audience
        self.key_for_token = key_for_token
        self.access = access or AccessMap()
        self.groups_claim, self.leeway = groups_claim, leeway

    @classmethod
    def from_env(cls) -> "TokenVerifier":
        issuer = os.environ["DOCBRIDGE_OIDC_ISSUER"].rstrip("/")
        jwks_url = os.environ.get("DOCBRIDGE_OIDC_JWKS_URL") or _discover_jwks(issuer)
        client = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=3600)
        return cls(issuer, os.environ["DOCBRIDGE_OIDC_AUDIENCE"],
                   lambda token: client.get_signing_key_from_jwt(token).key,
                   AccessMap.load(os.environ.get("DOCBRIDGE_ACCESS_FILE")),
                   os.environ.get("DOCBRIDGE_GROUPS_CLAIM", "groups"))

    def verify(self, token: str) -> Principal:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as e:
            raise AuthError("malformed token") from e
        if header.get("alg") not in ASYMMETRIC:
            raise AuthError(f"algorithm {header.get('alg')!r} is not accepted")
        try:
            claims = jwt.decode(token, self.key_for_token(token), algorithms=ASYMMETRIC, audience=self.audience,
                                issuer=self.issuer, leeway=self.leeway,
                                options={"require": ["exp", "iat", "iss", "aud", "sub"]})
        except jwt.PyJWTError as e:
            raise AuthError(str(e)) from e
        email = claims.get("email") if claims.get("email_verified", False) else None
        token_groups = claims.get(self.groups_claim, [])
        if isinstance(token_groups, str):
            token_groups = token_groups.split()
        groups = {str(g) for g in token_groups} | self.access.groups_for(email) | {"all"}
        return Principal(claims["sub"], email, tuple(sorted(groups)))


def _discover_jwks(issuer: str) -> str:
    import urllib.request

    with urllib.request.urlopen(f"{issuer}/.well-known/openid-configuration", timeout=10) as r:
        return json.load(r)["jwks_uri"]


class RateLimiter:
    """Token bucket per subject: `rate` requests per minute, bursts up to `burst`."""

    def __init__(self, rate: float = 30, burst: int = 10, clock: Callable[[], float] = time.monotonic):
        self.per_second, self.burst, self.clock = rate / 60.0, burst, clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.per_second)
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1, now)
            return True
