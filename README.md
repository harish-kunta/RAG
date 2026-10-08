# Learn RAG by building it

This project starts with a small retrieval system you can understand end to end. We will add one capability at a time: files, chunking, search, an answer model, then database and external-note connectors.

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

## Ask a question

```sh
python3 rag.py "What does the sample warranty cover?"
```

By default, the script retrieves up to three passages and asks `gpt-6-luna` to answer from them. To inspect retrieval without making an API request:

```sh
python3 rag.py "What does the sample warranty cover?" --context-only
```

The current OpenAI pricing page lists GPT-6 Luna at $0.10 per million input tokens and $0.50 per million output tokens for standard processing. Actual cost depends on tokens used, and pricing can change. See the [official OpenAI pricing page](https://developers.openai.com/api/docs/pricing).

## The data path

```text
Files → Documents → Chunks → Search → Retrieved context
```

- A **document** is a source item with text and metadata, such as its filename.
- A **chunk** is a manageable passage from one document. Search returns chunks, while metadata lets us identify their source.
- The first retriever is lexical: it looks for words shared by the question and each chunk. It is intentionally simple and will miss paraphrases.
- The answer model receives only the retrieved passages. It is instructed to cite them and say when they do not contain an answer.
- A later embedding retriever will search by semantic similarity; it can find related wording, but still depends on good content and chunking.

## Try it

Run the example question, then try `python3 rag.py "Can I send this back after a month?"`. That phrasing may retrieve nothing because the current search depends on matching words. The answer model cannot use evidence that retrieval did not find.

Next we will improve chunking and search, then connect more source types. Database rows and external notes can use the same document shape as files: each connector reads source-specific data and converts it into documents before indexing.
