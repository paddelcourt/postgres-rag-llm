### UV Python Initialization ###

Recommend to use uv python to initialize the project.

```bash
uv init
```

install the requirements

```bash
uv pip install -r requirements.txt
```

initialize the environment variables

```bash
uv venv
```


### LLM Model ###
This project uses Gemini as the LLM model. You can set the environment variable `GEMINI_API_KEY` in the `.env` file.

### Dataset ###
Dataset comes from Binaryy/multimodal-real-estate-search on Huggin Face. Check it out [here](https://huggingface.co/datasets/Binaryy/multimodal-real-estate-search).
For the purpose of this project, we are limiting the dataset to 100 rows. This is done on postgress-sql.ipynb file.


### 🐳 **Run Postgres in Docker**

- Install Docker Desktop from **[here](https://www.docker.com/products/docker-desktop/)**.
- Copy **`example.env`** to **`.env`**:
    
    ```bash
    cp example.env .env
    ```

- Start the Docker Compose container:
    - If you're on Mac:
        
        ```bash
        make up
        ```
        
    - If you're on Windows:
        
        ```bash
        docker compose up -d
        ```
        
- A folder named **`postgres-data`** will be created in the root of the repo. The data backing your Postgres instance will be saved here.
- You can check that your Docker Compose stack is running by either:
    - Going into Docker Desktop: you should see an entry there with a drop-down for each of the containers running in your Docker Compose stack.
    - Running **`docker ps -a`** and looking for the containers with the name **`postgres`**.


- When you're finished with your Postgres instance, you can stop the Docker Compose containers with:
    
    ```bash
    docker compose stop
    ```


### Setting up real estate data and postgres database ###

1. Download the real estate data above
2. Put the parquet file in the `data` folder
3. Run the postgres-sql.ipynb file to load the data into postgres and create the database tables
3. Run the rag-pipeline.py file to test the pipeline

## How the Code Works

This project implements a Retrieval Augmented Generation (RAG) pipeline for real estate data using PostgreSQL and the Gemini API. Here's how the system works:

### Architecture Overview

The RAG pipeline consists of the following main components:

1. **Data Extraction**: Reads real estate listings from a PostgreSQL database
2. **Text Embedding**: Generates vector embeddings for each property listing using Gemini's text-embedding-004 model
3. **Database Storage**: Stores embeddings alongside original data in PostgreSQL
4. **Search Functionality**: Provides hybrid search capabilities combining semantic similarity, text search, and fuzzy search
5. **Answer Generation**: Uses Gemini's generative AI to create natural language responses based on retrieved properties

### Key Files and Components

- **rag-pipeline.py**: Main script that orchestrates the entire RAG pipeline
- **utils.py**: Contains utility functions for database operations, embedding generation, and search
- **postgres-sql.ipynb**: Jupyter notebook for initial data setup and database exploration

### Pipeline Steps

1. **Initialization**: Connects to PostgreSQL and initializes the Gemini API client
2. **Data Ingestion**:
   - Reads property listings from the database
   - Generates embeddings in configurable batch sizes
   - Updates the database with embedding vectors
   - Sets up search indexes and functions
3. **Interactive Search**:
   - Takes user queries in natural language
   - Performs hybrid search using both semantic similarity and keyword matching
   - Retrieves relevant property listings
   - Generates natural language responses based on the retrieved data


## Running the Pipeline

### Command-Line Arguments

The RAG pipeline (`rag-pipeline.py`) supports several command-line arguments to customize execution:

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch=N` | Set the batch size for processing records | 10 |
| `--skip-embeddings` | Skip embedding generation (use dummy vectors) | False |
| `--skip-updates` | Skip database updates | False |
| `--skip-ingestion` | Skip data ingestion (search only) | False |
| `--help` | Show help message | - |

### Usage Examples

1. **Full Pipeline Run**:
   ```bash
   uv run rag-pipeline.py
   ```
   This runs the complete pipeline including data extraction, embedding generation, database updates, and interactive search.

2. **Search Only Mode**:
   ```bash
   uv run rag-pipeline.py --skip-ingestion
   ```
   Skips the data ingestion phase and directly starts the interactive search interface.

3. **Processing with Larger Batches**:
   ```bash
   uv run rag-pipeline.py --batch=20
   ```
   Processes records in batches of 20 instead of the default 10.

4. **Testing without API Costs**:
   ```bash
   uv run rag-pipeline.py --skip-embeddings
   ```
   Uses dummy embedding vectors (all zeros) to test the pipeline without making API calls.

5. **Testing without Database Updates**:
   ```bash
   uv run rag-pipeline.py --skip-updates
   ```
   Generates embeddings but doesn't update the database.

### Interactive Search Usage

Once the interactive search interface starts, you can:
- Enter natural language queries about real estate properties
- Ask about specific locations, property types, or features
- Type 'quit' or 'exit' to end the session

Example output:

```bash

Type your questions or 'quit' to exit

Your question: hi, can you find me 3 bedroom in lagos

Processing your query...
Enhanced search query: '3 bedroom Lagos'
Searching with enhanced keywords...
Search debug info: {'text': {'rank': 1, 'score': 0.12222222}, 'fuzzy': {'rank': 2, 'score': 0.32258064, 'title_sim': 0.32258064, 'location_sim': 0.16216215}, 'vector': {'rank': None, 'score': None}}
Found 5 relevant results

Answer: Yes, I can find you 3-bedroom properties in Lagos from the listings provided. Here's a summary:

*   **Chevron, Lekki Phase 2, Lekki, Lagos:** A 3-bedroom apartment is available.
*   **Lekki Phase 1, Lekki, Lagos:** Newly completed and exquisitely built 3-bedroom apartments are for sale in a serviced estate with amenities like a swimming pool and gym. The price is N150 million. 
*   **Ologolo, Lekki, Lagos:** Exquisitely designed 3-bedroom apartments are for sale with amenities like a swimming pool. The price is #65M.
*   **Orchid, Lekki, Lagos:** Luxury 3-bedroom terraced duplexes are for sale for N65,000,000.


```
