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


### 🐳 **Run Postgres and PGAdmin in Docker**

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
- If you navigate to **`http://localhost:5050`** you will be able to see the PGAdmin instance up and running and should be able to connect to the following server as details shown:
    
    <img src=".attachments/pgadmin-server.png" style="width:500px;"/> 


- When you're finished with your Postgres instance(required in week 1 & 2 & 4), you can stop the Docker Compose containers with:
    
    ```bash
    docker compose stop
    ```


### Setting up real estate data and postgres database###

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
4. **Search Functionality**: Provides hybrid search capabilities combining semantic similarity and keyword matching
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

### Search Mechanism

The system provides a sophisticated search mechanism with hybrid search based on Supabase's hybrid search.

- **Hybrid Search**: Combines both approaches for optimal results

The search functionality includes intelligent recognition of:
- Location-based queries (e.g., "Lagos", "Lekki")
- Property type queries (e.g., "apartment", "duplex")
- Bedroom count queries (e.g., "3 bedroom")

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

Example queries:
- "Can you find me an apartment in Lagos?"
- "Show me 3 bedroom properties in Lekki"
- "What properties are available in Port Harcourt?"




