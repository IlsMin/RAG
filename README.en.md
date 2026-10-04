# RAG — a Habr knowledge base

[По-русски](README.md)

A C++ / Qt 6 desktop app that answers from a local dump of Habr questions and shows each retrieval step next to the answer. Chat and embeddings go through two local [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` processes. No Ollama and no cloud API key.

## What the window shows

- **Dialog.** The question and the answer are on the left. The step log is on the right: multi-query lines, FAISS scores, full text, fusion, rerank, and the prompt sent to the model. Under the answer: elapsed time and the names of the modes that were on.
- **Settings.** Server URLs, a no-RAG mode, three retrieval techniques, how many candidates and chunks go into the prompt, and `questions.jsonl` import.

The SQLite database and the index file live in a `data` directory next to the executable.

## Servers

Two processes are required. One `llama-server` cannot serve both chat and embeddings.

| Port | Role | Example |
| --- | --- | --- |
| 8080 | Chat, an instruct model, no LoRA | `llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 -c 4096 -ngl 8` |
| 8081 | Embeddings, a dedicated embedding model | `llama-server -m user-bge-m3-q8_0.gguf --host 127.0.0.1 --port 8081 --embeddings --pooling cls -c 2048 -ngl 99` |

Port 8081 needs a model trained as an embedder. This project was checked with `USER-bge-m3` (1024 dimensions, `cls` pooling). A plain instruct model with `--pooling mean` barely separates chunks. After changing the embedder, clear the database and import again: the index dimension is fixed by the first vector.

## Modes

While RAG is on, the question is always embedded and searched with FAISS (`IndexFlatIP` over L2-normalized vectors). The three checkboxes add steps on top of that search.

- **Multi-query.** The model adds exactly two short queries in Russian. Lines without Cyrillic, or lines that contain CJK characters, are dropped.
- **Hybrid search.** An SQLite FTS5 list is merged with the vector list by Reciprocal Rank Fusion.
- **Rerank.** The model reorders the retrieved chunks.
- **No RAG.** The question is sent to port 8080 as-is, with no database and no embedding.

Import reads JSON Lines in the Habr shape (`title`, HTML in `description` and `answers[].body`), splits answers into chunks, and does not clear the database first. Importing the same questions again duplicates them. `questions.jsonl` is not in this repository.

## Build

Windows, CMake 3.21+, Qt 6.5 (Core, Gui, Widgets, Sql, Network), and MSVC with OpenMP. A subset of FAISS 1.7.4 and the OpenBLAS libraries are vendored under `third_party` and built with the app.

Open `CMakeLists.txt` in Qt Creator and build the `RAG` target. Or from a Visual Studio command prompt:

```bat
cmake -S . -B build -DCMAKE_PREFIX_PATH=C:/Qt/6.11.1/msvc2022_64
cmake --build build --config Release
```

Point `CMAKE_PREFIX_PATH` at your Qt kit. `openblas.dll` is copied next to the executable by a POST_BUILD step.
