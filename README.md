# Learn RAG by building it

This project builds a small retrieval system one concept at a time: files, chunking, lexical and semantic search, answer generation, then database and external-note connectors.

## Lesson 1: retrieve useful text

The pipeline reads text and Markdown files from `data/`, splits them into chunks, scores chunks against a question using word overlap, and sends the best matches to an OpenAI model to write an answer with source citations.

## Set up the OpenAI API

In a terminal opened at this project folder, install the small Python SDK dependency and export your key into the current shell session:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

Keep the key in your terminal environment; do not paste it into this repository or into chat. The script reads `OPENAI_API_KEY` automatically.

## Lesson 2: semantic search with embeddings

Lexical search compares words. Embedding search compares vectors that represent the meaning of text, so differently worded questions can still find related passages.

Build the vector index once after setting `OPENAI_API_KEY`:

```sh
python3 rag.py --index
```

This embeds each chunk with `text-embedding-3-small` and saves the vectors plus source text and metadata in `.rag_index.json`. The file is local and ignored by Git. Rebuild the index after adding, editing, or deleting source files.

Now ask a paraphrased question:

```sh
python3 rag.py "Can I send this back after a month?"
```

To inspect the comparison between word matching and semantic matching without calling the answer model:

```sh
python3 rag.py "Can I send this back after a month?" --compare --context-only
```

This comparison still sends the question to the embeddings API. For a fully local lexical-only inspection, use:

```sh
python3 rag.py "Can I send this back after a month?" --lexical --context-only
```

The official OpenAI docs list `text-embedding-3-small` at $0.02 per million input tokens. Source chunks are embedded once per index build; each semantic question needs one query embedding. [Embedding model details](https://developers.openai.com/api/docs/models/text-embedding-3-small) · [Embedding guide](https://developers.openai.com/api/docs/guides/embeddings)

## Ask a question

```sh
python3 rag.py "What does the sample warranty cover?"
```

By default, the script embeds the question, retrieves up to three semantically similar passages, and asks `gpt-6-luna` to answer from them. To inspect lexical retrieval without any API request:

```sh
python3 rag.py "What does the sample warranty cover?" --lexical --context-only
```

The current OpenAI pricing page lists GPT-6 Luna at $0.10 per million input tokens and $0.50 per million output tokens for standard processing. Actual cost depends on tokens used, and pricing can change. See the [official OpenAI pricing page](https://developers.openai.com/api/docs/pricing).

## The data path

```text
Files → Documents → Chunks → Embeddings → Vector index
Question → Embedding → Similarity search → Retrieved context → Answer model
```

- A **document** is a source item with text and metadata, such as its filename.
- A **chunk** is a manageable passage from one document. Search returns chunks, while metadata lets us identify their source.
- The lexical retriever looks for words shared by the question and each chunk. It is local and easy to inspect, but misses paraphrases.
- The semantic retriever embeds each passage once, embeds a question when asked, then ranks passage vectors by cosine similarity. The JSON index is a tiny teaching stand-in for a vector database.
- The answer model receives only the retrieved passages. It is instructed to cite them and say when they do not contain an answer.
- A changed source file makes the saved index stale; the program checks this and asks you to rebuild it.

## Try it

Run both the lexical and semantic commands for the same paraphrased question. Compare the retrieved source chunks before looking at the generated answer. This helps distinguish a retrieval issue from an answer-generation issue.

Next we will improve chunking and search, then connect more source types. Database rows and external notes can use the same document shape as files: each connector reads source-specific data and converts it into documents before indexing.
