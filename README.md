# GraphTool

GraphTool builds and searches a knowledge graph from files placed under
`documents/`. Markdown, PDF, PowerPoint, and audio inputs are discovered
recursively. A
recommended layout is:

```text
documents/
├── markdown/
├── pdfs/
├── presentations/
└── recordings/
```

The folders are organizational; supported files can be nested anywhere under
`documents/` and are identified by extension.

PDFs are rendered page by page and converted to Markdown with the configured
`AZURE_OPENAI_FAST_DEPLOYMENT`. Converted pages are cached under
`data/pdf_conversions/`, so unchanged PDFs do not require additional model calls.

PowerPoint `.pptx` files are converted to PDF with LibreOffice, one slide per
page, and then use the same text-and-image PDF conversion pipeline. Generated
PDFs are cached under `data/presentation_conversions/`; source references retain
the original `.pptx` path and display page ranges as slide ranges. Speaker notes,
comments, animations, and embedded media are not ingested.

## PDF and PowerPoint requirements

Install the project dependencies with `uv sync`. PDF rendering also requires
Poppler's `pdftoppm` executable, and PowerPoint conversion requires LibreOffice's
`soffice` executable. Audio transcription requires the `ffmpeg` and `ffprobe`
executables:

```sh
# macOS
brew install poppler
brew install --cask libreoffice
brew install ffmpeg

# Ubuntu or Debian
sudo apt-get install poppler-utils
sudo apt-get install libreoffice
sudo apt-get install ffmpeg
```

The configured fast Azure OpenAI deployment must support image input and
structured output. Password-protected PDFs are not supported.

## Audio transcription

GraphTool accepts `.flac`, `.m4a`, `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.ogg`,
`.wav`, and `.webm` files. Audio is normalized to mono 16 kHz MP3 at 64 kbps,
split into approximately 20-minute chunks with five seconds of overlap, and
transcribed sequentially with `AZURE_OPENAI_TRANSCRIPTION_DEPLOYMENT`.

Chunk transcripts and assembled Markdown are cached under
`data/audio_transcriptions/`. This directory is generated data; put original
recordings under `documents/`, preferably `documents/recordings/`. Interrupted
conversions resume from the last cached chunk. The configured deployment should
use `gpt-4o-transcribe`.

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
