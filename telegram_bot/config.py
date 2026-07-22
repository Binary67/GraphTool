import os
from dataclasses import dataclass

from dotenv import load_dotenv

BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
ALLOWED_USER_IDS_ENV = "TELEGRAM_ALLOWED_USER_IDS"


class TelegramBotConfigError(ValueError):
    """Raised when required Telegram bot configuration is invalid."""


@dataclass(frozen=True)
class TelegramBotConfig:
    token: str
    allowed_user_ids: frozenset[int]


def load_telegram_bot_config() -> TelegramBotConfig:
    load_dotenv(override=True)
    token = (os.getenv(BOT_TOKEN_ENV) or "").strip()
    allowed_user_ids_text = (os.getenv(ALLOWED_USER_IDS_ENV) or "").strip()

    missing = []
    if not token:
        missing.append(BOT_TOKEN_ENV)
    if not allowed_user_ids_text:
        missing.append(ALLOWED_USER_IDS_ENV)
    if missing:
        raise TelegramBotConfigError(
            f"Missing required Telegram environment variables: {', '.join(missing)}"
        )

    try:
        allowed_user_ids = frozenset(
            int(value.strip()) for value in allowed_user_ids_text.split(",")
        )
    except ValueError as exc:
        raise TelegramBotConfigError(
            f"{ALLOWED_USER_IDS_ENV} must contain comma-separated numeric user IDs"
        ) from exc
    if any(user_id <= 0 for user_id in allowed_user_ids):
        raise TelegramBotConfigError(
            f"{ALLOWED_USER_IDS_ENV} must contain positive numeric user IDs"
        )

    return TelegramBotConfig(
        token=token,
        allowed_user_ids=allowed_user_ids,
    )
