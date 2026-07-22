from unittest.mock import Mock

import pytest

from telegram_bot.config import (
    ALLOWED_USER_IDS_ENV,
    BOT_TOKEN_ENV,
    TelegramBotConfigError,
    load_telegram_bot_config,
)


def test_loads_telegram_bot_config(monkeypatch):
    load_dotenv = Mock()
    monkeypatch.setattr("telegram_bot.config.load_dotenv", load_dotenv)
    monkeypatch.setenv(BOT_TOKEN_ENV, " bot-token ")
    monkeypatch.setenv(ALLOWED_USER_IDS_ENV, "123, 456")

    config = load_telegram_bot_config()

    load_dotenv.assert_called_once_with(override=True)
    assert config.token == "bot-token"
    assert config.allowed_user_ids == frozenset({123, 456})


def test_missing_telegram_bot_config_raises_clear_error(monkeypatch):
    monkeypatch.setattr("telegram_bot.config.load_dotenv", Mock())
    monkeypatch.delenv(BOT_TOKEN_ENV, raising=False)
    monkeypatch.delenv(ALLOWED_USER_IDS_ENV, raising=False)

    with pytest.raises(TelegramBotConfigError) as exc_info:
        load_telegram_bot_config()

    message = str(exc_info.value)
    assert BOT_TOKEN_ENV in message
    assert ALLOWED_USER_IDS_ENV in message


@pytest.mark.parametrize("value", ["abc", "123,", "0", "-1"])
def test_rejects_invalid_allowed_user_ids(monkeypatch, value):
    monkeypatch.setattr("telegram_bot.config.load_dotenv", Mock())
    monkeypatch.setenv(BOT_TOKEN_ENV, "bot-token")
    monkeypatch.setenv(ALLOWED_USER_IDS_ENV, value)

    with pytest.raises(TelegramBotConfigError):
        load_telegram_bot_config()
