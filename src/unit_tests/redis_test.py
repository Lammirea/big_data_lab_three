import redis
import unittest
import os

try:
    import fakeredis
except ImportError:
    fakeredis = None

class TestRedisIntegration(unittest.TestCase):
    def setUp(self):
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_password = os.getenv('REDIS_PASSWORD', None)
        self.redis_db = int(os.getenv('REDIS_DB', 0))

        # Пытаемся подключиться к реальному Redis, иначе используем fakeredis (если установлен)
        real_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            db=self.redis_db,
            decode_responses=True
        )

        try:
            if real_client.ping():
                self.redis_client = real_client
                return
        except redis.exceptions.ConnectionError:
            pass

        if fakeredis is not None:
            # использовать fakeredis как drop-in замену
            self.redis_client = fakeredis.FakeStrictRedis(decode_responses=True)
        else:
            # если fakeredis не установлен, пропускаем тесты
            raise unittest.SkipTest("Redis недоступен и fakeredis не установлен")

    def test_redis_connection(self):
        self.assertTrue(self.redis_client.ping())

    def test_data_persistence(self):
        key = 'test_key'
        val = 'test_value'
        self.redis_client.delete(key)
        self.redis_client.set(key, val)
        result = self.redis_client.get(key)
        self.assertEqual(result, val)

if __name__ == '__main__':
    unittest.main()
