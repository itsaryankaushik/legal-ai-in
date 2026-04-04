# tests/unit/test_db_clients.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_redis_set_and_get_pageindex():
    from db.redis_client import RedisClient
    client = RedisClient.__new__(RedisClient)
    client._redis = AsyncMock()
    client._redis.set = AsyncMock(return_value=True)
    client._redis.get = AsyncMock(return_value=b'{"node_id": "BNS-103"}')

    await client.set_case_pageindex("CASE-001", {"node_id": "BNS-103"})
    result = await client.get_case_pageindex("CASE-001")

    assert result == {"node_id": "BNS-103"}
    client._redis.set.assert_called_once()


@pytest.mark.asyncio
async def test_redis_delete_case_pageindex():
    from db.redis_client import RedisClient
    client = RedisClient.__new__(RedisClient)
    client._redis = AsyncMock()
    client._redis.delete = AsyncMock(return_value=1)

    await client.invalidate_case_pageindex("CASE-001")
    client._redis.delete.assert_called_with("pageindex:case:CASE-001")


@pytest.mark.asyncio
async def test_redis_append_session_history():
    from db.redis_client import RedisClient
    client = RedisClient.__new__(RedisClient)
    client._redis = AsyncMock()
    client._redis.rpush = AsyncMock(return_value=1)
    client._redis.ltrim = AsyncMock(return_value=True)
    client._redis.expire = AsyncMock(return_value=True)

    msg = {"role": "user", "content": "What is Section 103 BNS?"}
    await client.append_session_message("SESSION-1", msg)
    client._redis.rpush.assert_called_once()


@pytest.mark.asyncio
async def test_redis_rate_limit_allows_under_limit():
    from db.redis_client import RedisClient
    client = RedisClient.__new__(RedisClient)
    client._redis = AsyncMock()
    client._redis.incr = AsyncMock(return_value=5)
    client._redis.expire = AsyncMock(return_value=True)

    result = await client.check_rate_limit("test-key", limit=60)
    assert result is True


@pytest.mark.asyncio
async def test_redis_rate_limit_blocks_over_limit():
    from db.redis_client import RedisClient
    client = RedisClient.__new__(RedisClient)
    client._redis = AsyncMock()
    client._redis.incr = AsyncMock(return_value=61)
    client._redis.expire = AsyncMock(return_value=True)

    result = await client.check_rate_limit("test-key", limit=60)
    assert result is False
