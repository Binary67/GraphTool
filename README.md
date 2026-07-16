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
