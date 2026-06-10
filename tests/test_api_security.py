import unittest
import sys
import asyncio
from pathlib import Path
from jose import jwt, JWTError
from datetime import timedelta, datetime, timezone
from fastapi import HTTPException, status

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.security import create_access_token, verify_admin_token, ALGORITHM
from configs.settings import settings

class TestApiSecurity(unittest.TestCase):
    def test_create_access_token_and_decode(self):
        """Verify that create_access_token signs correct payload with secret key."""
        payload = {"sub": "admin", "role": "hpe_storage_admin"}
        token = create_access_token(payload, expires_delta=timedelta(minutes=5))
        
        decoded = jwt.decode(token, settings.api_key_secret, algorithms=[ALGORITHM])
        self.assertEqual(decoded["sub"], "admin")
        self.assertEqual(decoded["role"], "hpe_storage_admin")
        self.assertTrue("exp" in decoded)

    def test_verify_admin_token_success(self):
        """Verify that verify_admin_token succeeds with correct credentials and role."""
        payload = {"sub": "admin", "role": "hpe_storage_admin"}
        token = create_access_token(payload)
        
        # verify_admin_token is an async function
        username = asyncio.run(verify_admin_token(token))
        self.assertEqual(username, "admin")

    def test_verify_admin_token_invalid_signature(self):
        """Verify that verify_admin_token raises 401 when token is signed with wrong key."""
        payload = {"sub": "admin", "role": "hpe_storage_admin"}
        # Signed with an arbitrary key
        bad_token = jwt.encode(payload, "wrong-secret-key", algorithm=ALGORITHM)
        
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(verify_admin_token(bad_token))
        self.assertEqual(ctx.exception.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_verify_admin_token_invalid_role(self):
        """Verify that verify_admin_token raises 401 if the role is not hpe_storage_admin."""
        payload = {"sub": "admin", "role": "regular_user"}
        token = create_access_token(payload)
        
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(verify_admin_token(token))
        self.assertEqual(ctx.exception.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_verify_admin_token_expired(self):
        """Verify that verify_admin_token raises 401 when the token is expired."""
        payload = {"sub": "admin", "role": "hpe_storage_admin"}
        # Generate token that expired 5 minutes ago
        token = create_access_token(payload, expires_delta=timedelta(minutes=-5))
        
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(verify_admin_token(token))
        self.assertEqual(ctx.exception.status_code, status.HTTP_401_UNAUTHORIZED)

if __name__ == "__main__":
    unittest.main()
