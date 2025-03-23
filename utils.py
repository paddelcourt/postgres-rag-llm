"""
Simplified RAG Database Utilities
---------------------------------
Core utilities for embedding generation, database operations, and search functionality.
"""

from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables and initialize API client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Constants
DB_NAME = "postgres"
DB_TABLE = "real_estate"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768

# PostgreSQL connection parameters
PG_PARAMS = {
    "dbname": DB_NAME,
    "user": "postgres",
    "password": "postgres",
    "host": "127.0.0.1",
    "port": "5432"
}

def get_connection():
    """Create and return a PostgreSQL connection."""
    return psycopg2.connect(**PG_PARAMS)

def generate_embedding(text):
    """Generate an embedding vector for the given text."""
    if not text or EMBEDDING_MODEL == "dummy":
        return [0.0] * EMBEDDING_DIM
    
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        return response.embeddings[0].values if response.embeddings else None
    except Exception as e:
        print(f"Embedding error: {str(e)[:100]}")
        return None

def setup_database():
    """Set up all necessary database structures."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Add vector extension and embedding column if needed
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            
            # Fix embedding column dimensions - drop and recreate if dimensions mismatch
            cur.execute(f"""
                DO $$ 
                BEGIN 
                    -- Drop existing index that depends on the embedding column
                    DROP INDEX IF EXISTS idx_embedding_hnsw;
                    
                    -- Drop and recreate embedding column to ensure correct dimensions
                    ALTER TABLE {DB_TABLE} DROP COLUMN IF EXISTS embedding;
                    ALTER TABLE {DB_TABLE} ADD COLUMN embedding vector({EMBEDDING_DIM});
                    RAISE NOTICE 'Reset embedding column to {EMBEDDING_DIM} dimensions';
                END $$;
            """)
            
            # Add tsvector column if it doesn't exist
            cur.execute(f"""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = '{DB_TABLE}' AND column_name = 'fts'
                    ) THEN
                        ALTER TABLE {DB_TABLE} ADD COLUMN fts tsvector;
                        RAISE NOTICE 'Added tsvector column';
                    END IF;
                END $$;
            """)
            
            # Update tsvector column with weighted content
            cur.execute(f"""
                UPDATE {DB_TABLE} 
                SET fts = setweight(to_tsvector('english', coalesce("Title", '')), 'A') || 
                          setweight(to_tsvector('english', coalesce("Location", '')), 'B') || 
                          setweight(to_tsvector('english', coalesce("Details", '')), 'C')
            """)
            
            # Create indexes
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_fts ON {DB_TABLE} USING GIN(fts)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_trigram_title ON {DB_TABLE} USING GIN (\"Title\" gin_trgm_ops)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_trigram_location ON {DB_TABLE} USING GIN (\"Location\" gin_trgm_ops)")
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_embedding_hnsw 
                ON {DB_TABLE} USING hnsw(embedding vector_ip_ops) 
                WITH (m=16, ef_construction=64)
            """)
            
            conn.commit()
            print("Database setup completed successfully")

def update_embeddings(data, column_names):
    """Update embeddings for the given data."""
    # Find column indices
    id_idx = column_names.index("id")
    title_idx = column_names.index("Title") if "Title" in column_names else 1
    location_idx = column_names.index("Location") if "Location" in column_names else 2
    details_idx = column_names.index("Details") if "Details" in column_names else 3
    
    updates = 0
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for row in data:
                # Generate embedding from combined text
                text = f"{row[title_idx]}. {row[location_idx]}. {row[details_idx]}"
                embedding = generate_embedding(text)
                
                if not embedding:
                    continue
                
                # Format embedding for PostgreSQL
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
                
                # Convert UUID to string if needed
                record_id = row[id_idx]
                record_id_str = str(record_id) if hasattr(record_id, 'hex') else record_id
                
                # Update record
                cur.execute(
                    f"UPDATE {DB_TABLE} SET embedding = %(embedding)s::vector WHERE id = %(id)s",
                    {"embedding": embedding_str, "id": record_id_str}
                )
                conn.commit()
                
                updates += 1
                if updates % 10 == 0:
                    print(f"Updated {updates} records")
    
    print(f"Updated {updates} records with embeddings")
    return updates

def hybrid_search(query_text, limit=5, text_weight=1.0, vector_weight=1.2):
    """Perform hybrid search using both vector similarity and text search."""
    # Generate embedding for the query
    query_embedding = generate_embedding(query_text)
    if not query_embedding:
        print("Could not generate embedding for query")
        return None
    
    # Format embedding for PostgreSQL
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    
    # RRF parameter (Reciprocal Rank Fusion)
    k_param = 60
    
    # SQL query using Common Table Expressions (CTEs) for hybrid search
    sql = """
    WITH semantic_search AS (
        SELECT id, "Title", "Location", "Details", 
               RANK () OVER (ORDER BY embedding <=> %(embedding)s) AS rank
        FROM """ + DB_TABLE + """
        ORDER BY embedding <=> %(embedding)s
        LIMIT 20
    ),
    keyword_search AS (
        SELECT id, "Title", "Location", "Details",
               RANK () OVER (ORDER BY ts_rank_cd(fts, to_tsquery('english', %(query)s)) DESC) AS rank
        FROM """ + DB_TABLE + """
        WHERE fts @@ to_tsquery('english', %(query)s)
        ORDER BY ts_rank_cd(fts, to_tsquery('english', %(query)s)) DESC
        LIMIT 20
    )
    SELECT
        COALESCE(s.id, k.id) AS id,
        COALESCE(s."Title", k."Title") AS "Title",
        COALESCE(s."Location", k."Location") AS "Location",
        COALESCE(s."Details", k."Details") AS "Details",
        (%(vector_weight)s * COALESCE(1.0 / (%(k)s + s.rank), 0.0)) +
        (%(text_weight)s * COALESCE(1.0 / (%(k)s + k.rank), 0.0)) AS score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY score DESC
    LIMIT %(limit)s
    """
    
    # Execute query with named parameters
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Convert spaces to & for tsquery
            ts_query = ' & '.join(query_text.split())
            
            # Execute with named parameters
            cur.execute(sql, {
                "embedding": embedding_str,
                "query": ts_query,
                "k": k_param,
                "vector_weight": float(vector_weight),
                "text_weight": float(text_weight),
                "limit": limit
            })
            
            # Get results
            results = cur.fetchall()
            
            if results:
                column_names = [desc[0] for desc in cur.description]
                return results, column_names
    
    return None

def get_all_data():
    """Get all data from the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {DB_TABLE}")
            data = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]
            return data, column_names
    
    return [], []




