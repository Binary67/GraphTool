# GraphTool

GraphTool builds and searches a knowledge graph from documents placed under
`documents/`. Markdown (`.md`) and PDF (`.pdf`) inputs are discovered
recursively.

PDFs are rendered page by page and converted to Markdown with the configured
`AZURE_OPENAI_FAST_DEPLOYMENT`. Converted pages are cached under
`data/pdf_conversions/`, so unchanged PDFs do not require additional model calls.

## PDF requirements

Install the project dependencies with `uv sync`. PDF rendering also requires
Poppler's `pdftoppm` executable:

```sh
# macOS
brew install poppler

# Ubuntu or Debian
sudo apt-get install poppler-utils
```

The configured fast Azure OpenAI deployment must support image input and
structured output. Password-protected PDFs are not supported.

## Knowledge agent

Set `AZURE_OPENAI_AGENT_DEPLOYMENT` to an Azure OpenAI deployment that supports
structured output. After synchronizing the knowledge base, create the read-only
agent through the Python API:

```python
from graphtool.agents import create_knowledge_agent
from graphtool.llm import (
    create_azure_openai_agent_model,
    load_azure_openai_config,
)
from graphtool.runtime import create_runtime

config = load_azure_openai_config()
runtime = create_runtime(config)
runtime.prepare_search()
model = create_azure_openai_agent_model(config)
agent = create_knowledge_agent(model, runtime)

response = agent.ask("What can GraphTool do?", thread_id="demo")
print(response.answer)
print(response.references)
```

Call `runtime.prepare_search()` after each document synchronization. It loads the
current knowledge base, prepares reusable search indexes, and refreshes stale chunk
embeddings before queries are served.

Calls using the same thread ID share conversation history while the process is
running. By default, older history is summarized when the conversation reaches
approximately 32,000 tokens, while the most recent 8,000 tokens remain verbatim.
These limits can be changed with `compact_trigger_tokens` and
`compact_recent_tokens` when creating the agent. The agent searches the local
knowledge base up to five times per question and returns `status="partial"` when
the available evidence remains incomplete.
