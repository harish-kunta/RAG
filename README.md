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

## Lesson 3: add database rows

The text files and team notes are one source. Now we add a small SQLite database as another source. The SQL schema and sample rows are readable in `data/database_seed.sql`.

Create the local demo database, then rebuild the vector index so it contains both files and database rows:

```sh
python3 rag.py --init-db
python3 rag.py --index
```

When reading the table, the SQLite connector turns each row into a `Document`: the title and body become searchable text, and fields such as record ID, team, and update date become metadata. A source label like `sqlite:support_articles:shipping-101` makes citations trace back to a specific record. The generated `.sqlite3` file is local and ignored by Git.

After changing database content, rebuild the vector index. The index fingerprint includes source text and metadata, so search detects changes and asks for a rebuild if the index is stale.

## Ask a question

```sh
python3 rag.py "What does the sample warranty cover?"
```

By default, the script embeds the question, retrieves up to three semantically similar passages, and asks `gpt-6-luna` to answer from them. To inspect lexical retrieval without any API request:

```sh
python3 rag.py "What does the sample warranty cover?" --lexical --context-only
```

The current OpenAI pricing page lists GPT-6 Luna at $0.10 per million input tokens and $0.50 per million output tokens for standard processing. Actual cost depends on tokens used, and pricing can change. See the [official OpenAI pricing page](https://developers.openai.com/api/docs/pricing).

## Lesson 4: add Notion notes

The Notion connector pulls page text into the same `Document` shape as files and SQLite rows. It uses Notion's API, so only content the integration can access is included.

1. Create an internal integration in Notion and grant it read access to content.
2. Copy its secret into your local terminal environment. Do not commit it or paste it into chat:

   ```sh
   export NOTION_API_KEY="your-integration-secret"
   ```

3. In Notion, share each page (or parent page) you want indexed with that integration.
4. Sync those pages and rebuild the vector index:

   ```sh
   python3 rag.py --sync-notion
   python3 rag.py --index
   ```

Now ask a question as usual. To inspect what retrieval found without calling the answer model:

```sh
python3 rag.py "What did we decide about onboarding?" --context-only
```

The sync saves readable page text and metadata in `data/notion_cache.json`; Git ignores this local cache. Sync again after Notion content changes, then rebuild the index. The starter connector extracts text blocks and nested blocks. It does not download files or interpret image contents, and it does not sync Notion databases as structured rows yet. The [Notion Search API](https://developers.notion.com/reference/post-search) finds pages available to the integration, and [block children](https://developers.notion.com/reference/get-block-children) provides page content. Both endpoints paginate, which the connector follows.

## Lesson 5: combine keyword and meaning search

Semantic search is good at paraphrases, while lexical search is useful for exact names, product codes, and phrases. Their raw scores have different scales, so adding those scores together would be misleading. Hybrid search combines their **rankings** with Reciprocal Rank Fusion (RRF): each result gets `1 / (60 + rank)` from each retriever, and results present in both lists get both contributions. This favors passages that rank well across both methods.

First inspect the two rankings separately, then inspect their fused ranking:

```sh
python3 rag.py "Does the warranty cover normal wear and tear?" --compare --context-only
python3 rag.py "Does the warranty cover normal wear and tear?" --hybrid --context-only
```

`--compare` displays lexical and semantic rankings side by side. `--hybrid` embeds the question once, like semantic search, and also runs local keyword search; it combines candidates from both lists, then uses the fused top passages for the answer. The displayed score is an RRF fusion score, not a probability or cosine similarity. Remove `--context-only` to generate an answer from those passages.

## The data path

```text
Files + SQLite rows + Notion pages → Documents → Chunks → Embeddings → Vector index
Question → Keyword search + embedding search → Rank fusion → Retrieved context → Answer model
```

- A **document** is a source item with text and metadata, such as its filename.
- A **chunk** is a manageable passage from one document. Search returns chunks, while metadata lets us identify their source.
- The lexical retriever looks for words shared by the question and each chunk. It is local and easy to inspect, but misses paraphrases.
- The semantic retriever embeds each passage once, embeds a question when asked, then ranks passage vectors by cosine similarity. The JSON index is a tiny teaching stand-in for a vector database.
- The hybrid retriever combines keyword and semantic result ranks with RRF. It helps when a question has exact terms and also uses wording different from the source.
- The answer model receives only the retrieved passages. It is instructed to cite them and say when they do not contain an answer.
- A changed source file makes the saved index stale; the program checks this and asks you to rebuild it.

## Try it

Run both the lexical and semantic commands for the same paraphrased question. Compare the retrieved source chunks before looking at the generated answer. This helps distinguish a retrieval issue from an answer-generation issue.

The sample notes remain a text file source. The Notion connector demonstrates how an external service can fetch records and convert them to the same `Document` shape before indexing.
