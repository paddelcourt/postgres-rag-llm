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

def hybrid_search(query_text, limit=5, text_weight=1.0, vector_weight=1.2, fuzzy_weight=0.8):
    """Perform hybrid search using vector similarity, text search, and fuzzy matching.
    
    This function uses a single SQL query with Common Table Expressions (CTEs) to:
    1. Perform vector similarity search
    2. Perform full-text search
    3. Perform fuzzy matching with similarity
    4. Combine all results using Reciprocal Rank Fusion
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query_text)
        if not query_embedding:
            print("Could not generate embedding for query")
            return None
        
        # Format embedding for PostgreSQL
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        
        # Prepare tsquery - replace spaces with '&' for AND logic
        try:
            ts_query = ' & '.join(word for word in query_text.split() if word)
        except Exception as e:
            print(f"Error preparing tsquery: {e}")
            ts_query = query_text
        
        # Pattern for ILIKE searches
        pattern = f"%{query_text}%"
        
        # RRF parameter (Reciprocal Rank Fusion)
        k_param = 60
        
        # SQL query using Common Table Expressions (CTEs) for hybrid search
        sql = f"""
        WITH 
        semantic_search AS (
            SELECT id, "Title", "Location", "Details", 
                   ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
            FROM {DB_TABLE}
            ORDER BY embedding <=> %s::vector
            LIMIT 30
        ),
        keyword_search AS (
            SELECT id, "Title", "Location", "Details",
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(fts, to_tsquery('english', %s)) DESC) AS rank
            FROM {DB_TABLE}
            WHERE fts @@ to_tsquery('english', %s)
            ORDER BY ts_rank_cd(fts, to_tsquery('english', %s)) DESC
            LIMIT 30
        ),
        fuzzy_search AS (
            SELECT id, "Title", "Location", "Details",
                   ROW_NUMBER() OVER (
                      ORDER BY 
                          greatest(
                              similarity("Title", %s),
                              similarity("Location", %s),
                              similarity("Details", %s)
                          ) DESC
                   ) AS rank
            FROM {DB_TABLE}
            WHERE "Title" ILIKE %s OR "Location" ILIKE %s OR "Details" ILIKE %s
            ORDER BY 
                greatest(
                    similarity("Title", %s),
                    similarity("Location", %s),
                    similarity("Details", %s)
                ) DESC
            LIMIT 30
        )
        SELECT
            COALESCE(s.id, COALESCE(k.id, f.id)) AS id,
            COALESCE(s."Title", COALESCE(k."Title", f."Title")) AS "Title",
            COALESCE(s."Location", COALESCE(k."Location", f."Location")) AS "Location",
            COALESCE(s."Details", COALESCE(k."Details", f."Details")) AS "Details",
            ((%s * COALESCE(1.0 / (%s + s.rank), 0.0)) +
             (%s * COALESCE(1.0 / (%s + k.rank), 0.0)) +
             (%s * COALESCE(1.0 / (%s + f.rank), 0.0))) AS score
        FROM semantic_search s
        FULL OUTER JOIN keyword_search k ON s.id = k.id
        FULL OUTER JOIN fuzzy_search f ON COALESCE(s.id, k.id) = f.id
        ORDER BY score DESC
        LIMIT %s
        """
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Ensure extensions are enabled for this session
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                try:
                    cur.execute(sql, (
                        embedding_str, embedding_str,                      # For semantic search 
                        ts_query, ts_query, ts_query,                      # For keyword search
                        query_text, query_text, query_text,                # For fuzzy search similarity
                        pattern, pattern, pattern,                         # For fuzzy search ILIKE conditions
                        query_text, query_text, query_text,                # For fuzzy search ORDER BY
                        vector_weight, k_param,                            # Vector weight and k param
                        text_weight, k_param,                              # Text weight and k param
                        fuzzy_weight, k_param,                             # Fuzzy weight and k param
                        limit                                              # Result limit
                    ))
                    
                    results = cur.fetchall()
                    if results:
                        column_names = [desc[0] for desc in cur.description]
                        print(f"Hybrid SQL query found {len(results)} results")
                        return results, column_names
                
                except Exception as e:
                    print(f"Hybrid query error: {e}")
                    
                    # Fall back to a simple search if hybrid query fails
                    print("Falling back to simple pattern matching...")
                    simple_sql = f"""
                    SELECT id, "Title", "Location", "Details"
                    FROM {DB_TABLE}
                    WHERE "Title" ILIKE %s OR "Location" ILIKE %s OR "Details" ILIKE %s
                    ORDER BY 
                        CASE WHEN "Title" ILIKE %s THEN 1
                             WHEN "Location" ILIKE %s THEN 2
                             WHEN "Details" ILIKE %s THEN 3
                             ELSE 4
                        END
                    LIMIT %s
                    """
                    
                    try:
                        cur.execute(simple_sql, (
                            pattern, pattern, pattern,  # For WHERE conditions
                            pattern, pattern, pattern,  # For ORDER BY conditions
                            limit
                        ))
                        
                        results = cur.fetchall()
                        if results:
                            column_names = ["id", "Title", "Location", "Details"]
                            print(f"Simple pattern search found {len(results)} results")
                            return results, column_names
                    except Exception as e:
                        print(f"Simple pattern search error: {e}")
    
    except Exception as e:
        print(f"Overall search error: {e}")
    
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




