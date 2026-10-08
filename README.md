# Learn RAG by building it

This project builds a small retrieval system one concept at a time: files, chunking, lexical and semantic search, answer generation, then database and external-note connectors. The CLI lessons implement the core steps directly so they are easy to inspect. The browser chat now uses LangChain integrations for chunking, embeddings, vector retrieval, and answer generation.

## Lesson 1: retrieve useful text

The pipeline reads text and Markdown files from `data/`, splits them into chunks, scores chunks against a question using word overlap, and sends the best matches to an OpenAI model to write an answer with source citations.

## Set up the OpenAI API

In a terminal opened at this project folder, create a Python 3.10+ environment, install the project dependencies, and export your key into the current shell session:

```sh
python3.12 --version
python3.12 -m venv .venv-langchain
source .venv-langchain/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

The `python3.12 --version` and `python --version` commands should report Python 3.12. If `python3.12` is not installed, install Python 3.10+ from the [official macOS downloads](https://www.python.org/downloads/macos/) and replace `python3.12` above with the installed version's command. The separate `.venv-langchain` directory avoids reusing an older Python 3.9 environment. Keep the key in your terminal environment; do not paste it into this repository or into chat. The scripts read `OPENAI_API_KEY` automatically.

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

## Lesson 10: use LangChain in the web chat

The source adapters still read files, SQLite, and Notion, then convert each source to a LangChain `Document`. The web path is:

```text
source adapters → LangChain Documents → RecursiveCharacterTextSplitter
  → OpenAIEmbeddings → persistent Chroma → LangChain retriever
  → keyword search + RRF → ChatPromptTemplate → ChatOpenAI → StrOutputParser
```

It uses LangChain's `RecursiveCharacterTextSplitter`, `OpenAIEmbeddings`, Chroma vector store/retriever, and prompt → chat model → output parser chain. It keeps the project's lexical search and Reciprocal Rank Fusion (RRF), so the web app combines keyword results with the LangChain semantic results.

Read the matching [text splitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter), [OpenAI embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/openai), [Chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma), and [ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai) guides alongside `langchain_rag.py`.

Create/activate the Python 3.10+ environment from the setup section, set `OPENAI_API_KEY` in the server terminal, create the sample database if needed, and build the LangChain index:

```sh
python3 rag.py --init-db
python3 langchain_rag.py --index
uvicorn chat_app:app --reload
```

Open <http://127.0.0.1:8000>. The first indexing run embeds the chunks and saves a persistent local Chroma database under `.langchain_chroma/`; query embeddings and answer generation use the OpenAI API. The index path is tied to a fingerprint of source contents and chunk settings. After sources change, run `python3 langchain_rag.py --index` again; use `--rebuild` to re-embed an unchanged source version.

The key stays in the server environment; the browser never receives it. Conversations now persist in `data/chat_history.sqlite3`: use the left-side list to reopen a chat, **New chat** to start another, and the × button to delete one. The database is created automatically when the server starts. This is still a local, single-user learning app: it has no login or per-user access controls, so anyone who can reach this instance can see its saved chats. Do not expose it publicly. Chroma is also local storage; see [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md) for the remaining production work.

This SQLite conversation store is a learning step toward a shared production database. It saves each user and assistant message together with the source cards shown for the answer. When a chat is reopened, the server loads its saved messages and uses the recent turns as context for follow-up questions. Delete a chat to remove its messages from this local database.

### Compare the web retriever with the from-scratch version

The labeled questions in `data/eval_questions.json` can score the retrieval used by the web chat beside the original hand-built retriever:

```sh
python3 rag.py --index
python3 langchain_rag.py --index
python3 langchain_rag.py --eval --top-k 3
```

Both indexes must match the current files, database, and synced notes. Re-run the two `--index` commands after changing sources. Evaluation embeds each question for both retrievers, but does not rebuild indexes or call the answer-generation model. It reports per-question results and averages for **Recall@3** (how many expected source documents appeared among the first three unique sources) and **MRR@3** (how high the first expected source appeared: rank 1 scores 1, rank 2 scores 0.5, rank 3 scores about 0.33, and missing from the first three scores 0).

The two configurations currently use different chunking: the from-scratch index uses 500 characters with no overlap, while the LangChain index uses 500 characters with 80 characters of overlap. So this first comparison tells you how the two complete setups perform; it does not isolate LangChain itself as the cause of a score difference. The evaluation set is also small, so use it to learn and catch regressions, not as proof of production quality.

To inspect answer quality too, each evaluation question has a human-written `reference_answer`. Run:

```sh
python3 langchain_rag.py --eval-answers
```

This runs retrieval and answer generation, then asks the configured chat model to score each answer from 0 (poor) to 2 (good) for correctness, groundedness in the retrieved passages, and citation quality. It makes one embedding request, one answer request, and one judge request per question. The judge is a helper, not ground truth: read the answer, reference, and judge note, and correct the labels if needed. This small demo does not yet measure how well the app abstains on questions its knowledge base cannot answer.

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
   python3 langchain_rag.py --index
   ```

The first index command refreshes the from-scratch CLI lesson index. The second refreshes the LangChain/Chroma index used by the web chat.

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

## Lesson 6: measure retrieval quality

Before changing a retriever, check whether it finds the sources you expect. `data/eval_questions.json` is a small evaluation set: each question has one or more relevant source IDs. These labels are examples you can edit as the knowledge base changes.

Create the sample SQLite database if you have not already, then build an index containing your current sources:

```sh
python3 rag.py --init-db
python3 rag.py --index
```

Compare all three retrieval methods:

```sh
python3 rag.py --eval --top-k 3
```

The report includes per-question **Recall@3** and **RR@3**, then their averages (**MRR@3** for reciprocal rank). Recall measures how many expected source documents appeared in the first three unique sources. Reciprocal rank gives more credit when the first relevant source appears near the top. Evaluation labels are at the document/source level, so multiple chunks from one document count as one source.

To evaluate lexical search locally without an OpenAI key or vector index:

```sh
python3 rag.py --eval --eval-method lexical --top-k 3
```

Semantic and hybrid evaluation use the existing vector index and embed each question, but do not call the answer-generation model. Rebuild the index if sources changed. This lesson measures retrieval only; it does not judge whether the generated answer is correct or supported by the retrieved text.

## Lesson 7: filter sources with metadata

Each source has metadata alongside its text. Files have `kind=file`; Notion pages have `kind=notion`; SQLite records have `kind=sqlite` and fields such as `team=fulfillment` or `team=support`. A filter narrows the chunks considered **before** ranking, which is useful when you know the kind of source or team that should answer the question.

Search just the fulfillment team’s SQLite records:

```sh
python3 rag.py "How long does domestic shipping take?" --kind sqlite --team fulfillment --lexical --context-only
```

You can filter only by source kind too, such as `--kind notion`, or only by team with `--team support`. Team matching ignores letter case. These filters work with lexical, semantic, hybrid, and evaluation commands. Search metadata filters help scope results, but they are not an access-control system: they do not decide who is allowed to read the underlying files, database, or Notion pages.

## Lesson 8: keep context across chunk boundaries

The current chunker groups text into passages of about 500 characters. With no overlap, a useful sentence can land at the end of one passage while its explanation starts in the next. Overlap repeats a small amount of the previous passage at the start of the next one, so either chunk may carry enough context to make sense on its own.

Try an 80-character overlap with the local lexical evaluation:

```sh
python3 rag.py --eval --eval-method lexical --chunk-overlap 80 --top-k 3
```

Compare its Recall@3 and MRR@3 with the zero-overlap baseline from Lesson 6. A higher score on these sample questions suggests the overlap helped this dataset; it does not guarantee improvement for every knowledge base. Overlap also repeats text, which can increase index size and embed cost.

For semantic or hybrid search in the from-scratch CLI, rebuild its vector index with the same overlap setting, then pass that setting when searching or evaluating:

```sh
python3 rag.py --index --chunk-overlap 80
python3 rag.py --eval --chunk-overlap 80 --top-k 3
```

The CLI index records its chunking setting. If you search with a different overlap, the program asks you to rebuild so each question is compared against vectors for the same chunks. The LangChain web app has its own 500-character/80-character-overlap setting and Chroma index. Overlap is measured in characters here; production systems usually tune chunk size and overlap against their own data and evaluation questions.

## Lesson 9: pack context for the answer model

Retrieval can return more useful passages than we want to send to the answer model. Context packing walks passages in retrieval order, skips exact duplicates, and includes the highest-ranked passages that fit a character budget. Each included passage keeps its source label for citations.

Ask for more chunks than the answer should receive, then cap the context sent to the model:

```sh
python3 rag.py "What do I need to know about a warranty claim?" --top-k 8 --context-chars 1600
```

The response reports how many passages and characters it included. This project uses characters as a simple teaching limit; model context windows are measured in tokens, and different text can use different numbers of tokens per character. A real application should budget with the tokenizer for its model. If the budget is too small to fit any passage, the program asks you to increase it.

## The data path

```text
Files + SQLite rows + Notion pages → Documents + metadata → Chunks → Embeddings → Vector index
Question → Metadata filters → Keyword search + embedding search → Rank fusion → Context packing → Answer model
```

- A **document** is a source item with text and metadata, such as its filename.
- **Metadata filters** narrow the searchable chunks by fields such as source kind or SQLite team before retrieval ranks them.
- A **chunk** is a manageable passage from one document. Search returns chunks, while metadata lets us identify their source.
- The lexical retriever looks for words shared by the question and each chunk. It is local and easy to inspect, but misses paraphrases.
- The semantic retriever embeds each passage once, embeds a question when asked, then ranks passage vectors by cosine similarity. The JSON index is a tiny teaching stand-in for a vector database.
- The hybrid retriever combines keyword and semantic result ranks with RRF. It helps when a question has exact terms and also uses wording different from the source.
- The evaluation set records which source documents should answer sample questions. Recall@k and MRR@k help compare retrieval methods before assessing answer quality.
- Chunk overlap repeats a little preceding text at each boundary so relevant context is less likely to be split away from the passage that needs it.
- Context packing limits how much retrieved text reaches the answer model while keeping source labels attached for citations.
- The answer model receives only the retrieved passages. It is instructed to cite them and say when they do not contain an answer.
- A changed source file makes the saved index stale; the program checks this and asks you to rebuild it.

## Try it

Run both the lexical and semantic commands for the same paraphrased question. Compare the retrieved source chunks before looking at the generated answer. This helps distinguish a retrieval issue from an answer-generation issue.

The sample notes remain a text file source. The Notion connector demonstrates how an external service can fetch records and convert them to the same `Document` shape before indexing.
