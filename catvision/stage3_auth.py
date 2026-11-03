"""Password hashing and short-lived, signed application tokens."""
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets

import jwt


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode('utf-8'), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return f'scrypt${salt}${digest}'


def verify_password(password: str, stored: str) -> bool:
    try:
        method, salt, expected = stored.split('$')
        if method != 'scrypt':
            return False
        digest = hashlib.scrypt(password.encode('utf-8'), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
        return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError):
        return False


def create_token(user_id: str, settings) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({'sub':user_id, 'iat':now, 'exp':now+timedelta(hours=settings.token_hours),
                       'iss':'catvision', 'aud':'catvision-web'}, settings.jwt_secret, algorithm='HS256')


def token_user(token: str, settings) -> str:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=['HS256'], audience='catvision-web',
                         issuer='catvision', options={'require':['sub','exp','iat']})
    if not isinstance(payload['sub'], str):
        raise jwt.InvalidTokenError('invalid subject')
    return payload['sub']
