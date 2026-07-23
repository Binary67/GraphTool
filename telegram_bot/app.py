from telegram.ext import Application, CommandHandler, MessageHandler, filters

from graphtool.agents import KnowledgeAgent, create_knowledge_agent
from graphtool.llm import (
    AGENT_MAX_RETRIES,
    AGENT_REQUEST_TIMEOUT_SECONDS,
    create_azure_openai_agent_model,
    load_azure_openai_config,
)
from graphtool.run_logging import configure_run_logger
from graphtool.runtime import DEFAULT_MAX_LOG_FILES, create_runtime, default_paths
from telegram_bot.config import TelegramBotConfig, load_telegram_bot_config
from telegram_bot.handlers import TelegramHandlers


def create_telegram_application(
    agent: KnowledgeAgent,
    config: TelegramBotConfig,
) -> Application:
    handlers = TelegramHandlers(agent, config.allowed_user_ids)
    application = Application.builder().token(config.token).build()
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("new", handlers.new_conversation))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.message)
    )
    return application


def run_telegram_bot() -> None:
    paths = default_paths()
    logger = configure_run_logger(paths.logs_dir, DEFAULT_MAX_LOG_FILES)
    logger.info("Starting GraphTool Telegram bot")
    logger.info(
        "Azure agent requests: timeout=%ds, retries=%d",
        AGENT_REQUEST_TIMEOUT_SECONDS,
        AGENT_MAX_RETRIES,
    )

    try:
        telegram_config = load_telegram_bot_config()
        azure_config = load_azure_openai_config()
        runtime = create_runtime(azure_config, paths=paths)
        runtime.prepare_search()
        model = create_azure_openai_agent_model(azure_config)
        agent = create_knowledge_agent(model, runtime)
        application = create_telegram_application(agent, telegram_config)
        application.run_polling()
    except Exception:
        logger.exception("Telegram bot run failed")
        raise
