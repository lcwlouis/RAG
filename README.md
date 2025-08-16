# RAG

Simple, modular Retrieval-Augmented Generation (RAG) with Context Cleaning Proof of Concept

## Objective

This project demonstrates a pipeline for Retrieval-Augmented Generation (RAG) with a focus on context cleaning. The goal is to finetune a small LLM to filter and clean retrieved document chunks before answer generation, improving both answer quality and latency.

## Repository Structure

``` file_dir
├── chroma_db/                # Chroma vector DB for document storage
├── chroma_db_test/           # Test Chroma DB
├── ingestion.py              # Script for ingesting documents into Chroma
├── main.py                   # Entry point (demo)
├── rag_pipeline.py           # Main RAG pipeline (retrieval, context cleaning, answer gen)
├── generate_queries.ipynb    # Notebook for generating test queries
├── testRAGApp.ipynb          # Notebook for testing the RAG pipeline
├── notebooks/                # Additional notebooks for query generation
├── tests/                    # Test cases and evaluation data
├── texts/                    # Source documents (txt, pdf)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata and dependencies
└── ...
```


## Environment Variables

Before running the pipeline, set up the following environment variables (e.g., in a `.env` file):

- `GOOGLE_API_KEY`: Required for Gemini/Google LLM integration.

Example `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> **Note:** If you are running the project with `uv` or `uvicorn`, API endpoints are not yet implemented. The current pipeline is CLI and notebook-based. FastAPI/uvicorn support is planned (see Next Steps).

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Ingest documents

Place your `.txt` or `.pdf` files in the `texts/` directory. Then run:

```bash
python ingestion.py --ingest_directory ./texts
```

### 3. Run the RAG pipeline

```bash
python rag_pipeline.py
```

### 4. Generate and test queries

- Use `generate_queries.ipynb` and notebooks in `notebooks/` to create and test queries.
- Use `testRAGApp.ipynb` for end-to-end testing and evaluation.

## Current Features

- Document ingestion and chunking with LangChain and ChromaDB
- Embedding with Ollama
- Retrieval and context cleaning pipeline
- Example notebooks for query/test generation
- Test cases for evaluation

## Dataset Creation Pipeline

### Citation Format

Citations should be as concise as possible—ideally a direct quote or a short paragraph—that allows the query to be answered completely and accurately. The goal of the context cleaner is to minimize noise and only pass the most relevant supporting text to the final answer-generating LLM.

This project is also aiming to create a high-quality dataset to fine tune LLMs for context-cleaning to be deployed in RAG systems. The process is as follows:

### Process Overview

1. **Document Parsing & Query Generation**
    - Parse the entire document (e.g., `.txt`, `.pdf`) through a large context LLM (e.g., GPT-5, Gemini 2.5 Pro).
    - Generate a diverse set of queries, expected answers, and citations (quotes/locations supporting the answer) for each document.
    - Example output schema:

    ```json
    {
      "query": "What is the main theme of Chapter 2?",
      "expected_answer": "The main theme is...",
      "citations": ["Chapter 2, Paragraph 3", "Quote: ..."]
    }
    ```

2. **Organic Retrieval Loop**
    - For each query, use the RAG pipeline to retrieve chunks as would happen in production.
    - Pass the retrieved chunks and the query to a cheaper LLM to check if the required context is present.
    - If the context is insufficient, the LLM should output a special tag (e.g., `NEED_MORE_CONTEXT`) to signal the pipeline to retrieve more/different chunks and try again.
    - Repeat until the context is sufficient or a maximum retrieval limit is reached.

3. **Output Structure**
    - Each dataset entry should include:
        - `query`: The question.
        - `expected_answer`: The answer from the full document.
        - `citations`: Supporting evidence.
        - `retrieved_chunks`: The actual chunks retrieved by the pipeline.
        - `context_sufficiency`: One of `"sufficient"`, `"insufficient"`, or `"NEED_MORE_CONTEXT"`.
        - `notes`: (Optional) Any additional comments or tags.

    Example:

    ```json
    {
      "query": "...",
      "expected_answer": "...",
      "citations": ["...", "..."],
      "retrieved_chunks": ["chunk1", "chunk2"],
      "context_sufficiency": "NEED_MORE_CONTEXT",
      "notes": "Chunk did not contain the required quote."
    }
    ```

4. **Tagging and Feedback**
    - The output structure allows the main LLM or pipeline to detect when more retrieval is needed and to loop accordingly.
    - This enables robust evaluation and training for context cleaning and retrieval strategies.

### Model Plans

The planned model for fine-tuning is either the Base PT version of Gemma3 1B or 270M, depending on resource constraints and performance.

The model is going to be a very small model therefore we have to keep the outputs as simple as possible.

### Training Dataset Generation requirements

1. **Structure**
   - The training dataset will consist of pairs of input and output:
     - **Input**: The user query and the retrieved document chunks.
     - **Output**: The cleaned citations from the document and the context sufficiency label.

    Example (sufficient):

    ```json
    {
        "input": {
          "query": "What is the main theme of Chapter 2?",
          "citations": ["Chapter 2, Paragraph 3", "Quote: ..."]
        },
        "output": {
          "cleaned_citations": ["Chapter 2, Paragraph 3", "Quote: ..."],
          "context_sufficiency": "SUFFICIENT"
        }
    }
    ```

    Example (insufficient):

    ```json
    {
        "input": {
          "query": "What is the main theme of Chapter 2?",
          "citations": ["Chapter 2, Paragraph 3", "Quote: ..."]
        },
        "output": {
          "cleaned_citations": ["Chapter 2"],
          "context_sufficiency": "NEED_MORE_CONTEXT"
        }
    }
    ```

2. **Collection**
   - Collect a diverse set of queries, expected answers, and citations from various documents.
   - Ensure a mix of easy and challenging examples to improve model robustness.

## Next Steps

1. **Dataset Expansion & Automation**

    - Expand the diversity of source documents (add more formats, domains).
    - Automate query and answer/citation generation using large-context and smart LLMs.
    - Implement the organic retrieval and context sufficiency loop as described above.
    - Script the creation of (query, chunk, citation, sufficiency) datasets for training.

2. **Evaluation & Metrics**

    - Implement automated evaluation scripts for context cleaning quality.
    - Add benchmarks and metrics (precision, recall, latency, etc).
    - Integrate test cases in `tests/` for regression testing.

3. **Model Fine-tuning**

    - Fine-tune a small LLM on the generated dataset for context cleaning.
    - Compare performance with/without context cleaning.

4. **Pipeline Improvements**

    - Add support for more file types (PDF, HTML, etc).
    - Improve chunking and retrieval strategies.
    - Add API endpoints (e.g., with FastAPI) for serving the pipeline.

5. **Documentation & Usability**

    - Expand this README with usage examples and troubleshooting.
    - Add docstrings and comments to all scripts.
    - Provide example outputs and expected results.

## Things to Consider

1. **Confidence**: Do we want the small model to provide a confidence level instead of a tag of sufficiency? (use Likert if so)

## Limitations

- PDF files are not directly supported yet; only plain text (`.txt`) files are ingested and processed at this stage.

## Contributing

Contributions are welcome! Please open issues or pull requests for suggestions, bugfixes, or new features.

## License

See [LICENSE](LICENSE).
