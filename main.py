from graphtool.agents import KnowledgeAgent, create_knowledge_agent
from graphtool.llm import (
    create_azure_openai_agent_model,
    create_azure_openai_fast_agent_model,
    load_azure_openai_config,
)
from graphtool.retrieval import SourceReference, format_source_reference
from graphtool.run_logging import configure_run_logger
from graphtool.runtime import DEFAULT_MAX_LOG_FILES, create_runtime, default_paths

TERMINAL_THREAD_ID = "terminal"
EXIT_COMMANDS = {"exit", "quit"}


def _format_source_reference(reference: SourceReference) -> str:
    return format_source_reference(reference)


def _run_conversation(agent: KnowledgeAgent) -> None:
    print("GraphTool agent ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.casefold() in EXIT_COMMANDS:
            print("Goodbye.")
            return

        response = agent.ask(question, thread_id=TERMINAL_THREAD_ID)
        label = "Agent (partial)" if response.status == "partial" else "Agent"
        print(f"\n{label}: {response.answer}")
        references = ", ".join(
            _format_source_reference(reference)
            for reference in response.references
        )
        print(f"Sources: {references or 'None'}")


def main() -> None:
    paths = default_paths()
    logger = configure_run_logger(paths.logs_dir, DEFAULT_MAX_LOG_FILES)
    logger.info("Started GraphTool agent")

    try:
        config = load_azure_openai_config()
        runtime = create_runtime(config, paths=paths)
        runtime.prepare_search()
        answer_model = create_azure_openai_agent_model(config)
        orchestration_model = create_azure_openai_fast_agent_model(config)
        agent = create_knowledge_agent(
            answer_model,
            orchestration_model,
            runtime,
        )
        _run_conversation(agent)

        logger.info("Finished GraphTool agent")
    except Exception:
        logger.exception("Agent run failed")
        raise


if __name__ == "__main__":
    main()
