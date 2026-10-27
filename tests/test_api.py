import json
import logging
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from docbridge.auth import AccessMap, AuthError, RateLimiter, TokenVerifier  # noqa: E402

ISS, AUD = "https://idp.example", "docbridge-test"


@pytest.fixture(scope="module")
def keys():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, other


def make_token(private, **overrides):
    now = int(time.time())
    claims = {"iss": ISS, "aud": AUD, "sub": "user-1", "iat": now, "exp": now + 600,
              "email": "alice@harbourline.example", "email_verified": True}
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, private, algorithm="RS256", headers={"kid": "k1"})


@pytest.fixture(scope="module")
def verifier(keys):
    access = AccessMap(users={"hr.lead@harbourline.example": ["hr"]}, domains={"harbourline.example": ["staff"]},
                       admins=["ops@harbourline.example"])
    return TokenVerifier(ISS, AUD, lambda token: keys[0].public_key(), access)


def test_valid_token_gives_groups_from_claim_and_access_map(keys, verifier):
    p = verifier.verify(make_token(keys[0], groups=["engineering"]))
    assert p.subject == "user-1"
    assert set(p.groups) == {"all", "staff", "engineering"}
    p = verifier.verify(make_token(keys[0], email="hr.lead@harbourline.example"))
    assert "hr" in p.groups


def test_unverified_email_is_not_used_for_groups(keys, verifier):
    p = verifier.verify(make_token(keys[0], email="hr.lead@harbourline.example", email_verified=False))
    assert p.groups == ("all",)


@pytest.mark.parametrize("change, message", [
    ({"exp": int(time.time()) - 120}, "(?i)expired"),
    ({"aud": "someone-else"}, "(?i)audience"),
    ({"iss": "https://evil.example"}, "(?i)issuer"),
    ({"sub": None}, "sub"),
    ({"exp": None}, "exp"),
])
def test_bad_claims_are_rejected(keys, verifier, change, message):
    with pytest.raises(AuthError, match=message):
        verifier.verify(make_token(keys[0], **change))


def test_wrong_key_and_tampering_are_rejected(keys, verifier):
    with pytest.raises(AuthError):
        verifier.verify(make_token(keys[1]))
    header, payload, sig = make_token(keys[0]).split(".")
    forged = jwt.utils.base64url_encode(json.dumps({"iss": ISS, "aud": AUD, "sub": "admin", "iat": 1,
                                                    "exp": 9999999999, "groups": ["hr"]}).encode()).decode()
    with pytest.raises(AuthError):
        verifier.verify(f"{header}.{forged}.{sig}")


def test_alg_none_and_hs256_confusion_are_refused(keys, verifier):
    now = int(time.time())
    claims = {"iss": ISS, "aud": AUD, "sub": "x", "iat": now, "exp": now + 60, "groups": ["hr"]}
    unsigned = jwt.encode(claims, None, algorithm="none")
    with pytest.raises(AuthError, match="not accepted"):
        verifier.verify(unsigned)
    # The classic confusion attack: sign HS256 using the public key as the HMAC secret.
    pem = keys[0].public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    header = jwt.utils.base64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = jwt.utils.base64url_encode(json.dumps(claims).encode())
    import hashlib
    import hmac
    sig = jwt.utils.base64url_encode(hmac.new(pem, header + b"." + body, hashlib.sha256).digest())
    with pytest.raises(AuthError, match="not accepted"):
        verifier.verify((header + b"." + body + b"." + sig).decode())


def test_rate_limiter_refills():
    now = [0.0]
    rl = RateLimiter(rate=60, burst=2, clock=lambda: now[0])
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a")
    assert rl.allow("b")  # buckets are per subject
    now[0] += 1.0  # 60 per minute is one per second
    assert rl.allow("a") and not rl.allow("a")


@pytest.fixture(scope="module")
def client(keys, verifier):
    from docbridge.api import create_app

    app = create_app(verifier=verifier, limiter=RateLimiter(rate=600, burst=50))
    return TestClient(app)


def auth(keys, **claims):
    return {"Authorization": "Bearer " + make_token(keys[0], **claims)}


def test_api_requires_sign_in(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 401
    r = client.post("/ask", json={"question": "hi"}, headers={"Authorization": "Bearer nonsense"})
    assert r.status_code == 401 and "invalid token" in r.json()["detail"]


def test_api_answers_with_citations_and_enforces_groups(client, keys, caplog):
    q = {"question": "What is the merit budget for this salary review?"}
    with caplog.at_level(logging.INFO, logger="docbridge.audit"):
        staff = client.post("/ask", json=q, headers=auth(keys)).json()
        hr = client.post("/ask", json=q, headers=auth(keys, email="hr.lead@harbourline.example")).json()
    assert not staff["answered"]
    assert hr["answered"] and "4.5 percent" in hr["answer"]
    assert hr["citations"] == ["compensation_guidelines.pdf#p1"]
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "docbridge.audit"]
    assert [l["answered"] for l in lines[-2:]] == [False, True]
    assert all("user-1" not in json.dumps(l) for l in lines)  # subjects are hashed in the audit log


def test_me_and_admin_only_ingest(client, keys):
    assert client.get("/me", headers=auth(keys)).json()["groups"] == ["all", "staff"]
    assert client.post("/admin/ingest", headers=auth(keys)).status_code == 403
    r = client.post("/admin/ingest", headers=auth(keys, email="ops@harbourline.example"))
    assert r.status_code == 200 and r.json()["skipped_unchanged"] == r.json()["chunks"]


def test_rate_limit_returns_429(keys, verifier):
    from docbridge.api import create_app

    c = TestClient(create_app(verifier=verifier, limiter=RateLimiter(rate=1, burst=2)))
    codes = [c.post("/ask", json={"question": "hotel cap?"}, headers=auth(keys)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_web_app_and_question_limits(client, keys):
    assert "textContent" in client.get("/").text
    assert client.post("/ask", json={"question": ""}, headers=auth(keys)).status_code == 422
    assert client.post("/ask", json={"question": "x" * 1001}, headers=auth(keys)).status_code == 422
