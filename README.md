# CodeCompass: agentic RAG copilot for any codebase

[![tests](https://github.com/manikantaswamy7026-max/codecompass/actions/workflows/ci.yml/badge.svg)](https://github.com/manikantaswamy7026-max/codecompass/actions)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Author:** Manikanta A M · [GitHub](https://github.com/manikantaswamy7026-max) · Live demo UI: `docs/index.html` (enable GitHub Pages)

Ask questions about a repository and get answers whose **citations are verified**:
every `path:start-end` in the answer is checked to exist and to have actually been
read or retrieved by the agent. Runs fully local with Ollama (stdlib HTTP client,
no cloud keys).

```
repo ──► syntax-aware chunks ──► BM25 + embeddings ──► RRF fusion ──► rerank ──► call-graph expansion
                                                                                       │
question ──► agent loop (search_code · read_file · find_references · go_to_definition · list_dir)
                                                                                       ▼
                                                                answer + verified citations (SSE stream)
```

## Quickstart
```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b && ollama pull nomic-embed-text

python -m codecompass.cli index  /path/to/repo
python -m codecompass.cli search /path/to/repo "where is auth handled" --mode hybrid+graph
python -m codecompass.cli ask    /path/to/repo "what breaks if I change hash_password?"

uvicorn codecompass.api:app --port 8000      # GET /ask streams agent steps as SSE
```
Without Ollama, indexing and keyword search still work (embeddings are skipped).

## Design notes
- **Chunking:** Python via `ast` (functions, methods, class headers, module code); other
  languages via boundary regex then line windows. Chunks carry the set of called names.
- **Retrieval:** BM25 with identifier splitting (`getUserName` → `get`, `user`, `name`) + vector
  search, fused with reciprocal rank fusion, then a small name/path-overlap rerank.
- **Graph expansion:** top hits pull in their callees/callers (name-based call graph, approximate).
- **Incremental:** SHA-1 per file; embeddings cached by chunk hash.
- **Agent protocol:** one JSON object per turn, so small local models without native tool calling work.
  Malformed replies are retried; a step budget forces a best-effort answer.
- **Safety:** `read_file` cannot escape the repo root; tool errors never crash the loop.

## Evaluation
```bash
python eval/run_eval.py --root . -k 5      # add your own questions in eval/questions.jsonl
```
Strict metric: when a question names `expect_symbols`, the retrieved chunk must *be* that symbol.
First run on this repo itself (12 questions, keyword-only because no Ollama was available):

| mode | hit@5 | MRR |
|---|---|---|
| bm25 | 0.58 | 0.41 |
| hybrid (rerank, no vectors) | 0.75 | 0.60 |
| hybrid + graph | 0.75 | 0.61 |

Caveats: tiny question set, same repo the heuristics were developed on (a small module-chunk
penalty was added after seeing early results), and the vector half is **not yet measured**.
Run it on 2–3 unfamiliar open-source repos with ~30 questions each before quoting numbers.
Graph expansion helps or hurts depending on k; report both.

## Tests
`python -m unittest discover -s tests` (chunker, tokenizer, search, incremental index, tools,
agent loop with a scripted fake LLM, citation verification).

## Status and roadmap
Built and unit-tested offline. **Not yet exercised against a live Ollama model or the FastAPI
server**, so expect to tune the prompt for your model.
- [ ] tree-sitter chunking + real symbol resolution for JS/TS/Java/Go
- [ ] cross-encoder reranker, LLM-judged answer correctness in the eval
- [ ] Next.js UI consuming the SSE stream (animated agent steps and code graph)
- [ ] `run_tests` tool, git blame/log tools, change-impact mode

## Credits and license
Created by **Manikanta A M**. Released under the MIT License (see `LICENSE` and `NOTICE`).
If you reuse this code, keep the copyright notice and link back to this repository.
