"""
Simplified RAG Database Utilities
---------------------------------
Core utilities for embedding generation, database operations, and search functionality.
This version is domain-agnostic and can be used with any text data.
"""

from google import genai
from google.genai import types
import duckdb
import os
from dotenv import load_dotenv
import psycopg2

# Initialize the Gemini client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Constants
DB_TABLE = "postgres.real_estate"
POSTGRES_TABLE = "real_estate"  # Table name without schema for direct PostgreSQL queries
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768

# PostgreSQL connection parameters
PG_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
    "host": "127.0.0.1",
    "port": "5432"
}

# =====================================================================
# DATA EXTRACTION AND EMBEDDING
# =====================================================================

def extract_data_from_source():
    """Extract data from the database table."""
    try:
        result = duckdb.sql(f"SELECT * FROM {DB_TABLE}")
        column_names = [col[0] for col in result.description]
        data_rows = result.fetchall()
        print(f"Extracted {len(data_rows)} records from database")
        return (data_rows, column_names)
    except Exception as e:
        print(f"Error extracting data: {e}")
        return ([], [])

def generate_embedding(text):
    """Generate an embedding for the given text using Gemini API."""
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        
        if response and response.embeddings and response.embeddings[0].values:
            return response.embeddings[0].values
        
        print("Warning: Invalid embedding response")
        return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def embed_data(data_tuple):
    """Generate embeddings for the data."""
    data_rows, column_names = data_tuple
    
    try:
        # Find column indices
        id_idx = column_names.index('id')
        title_idx = column_names.index('Title')
        location_idx = column_names.index('Location')
        details_idx = column_names.index('Details')
        
        embeddings = []
        for row in data_rows:
            # Combine text fields for embedding
            text = f"{row[title_idx]}. {row[location_idx]}. {row[details_idx]}"
            embedding = generate_embedding(text)
            
            if embedding:
                embeddings.append(embedding)
            else:
                # Use dummy embedding if generation fails
                embeddings.append([0.0] * EMBEDDING_DIM)
                
        return embeddings
    except Exception as e:
        print(f"Error in embed_data: {e}")
        return [[0.0] * EMBEDDING_DIM for _ in range(len(data_rows))]

# =====================================================================
# DATABASE OPERATIONS
# =====================================================================

def add_embedding_column():
    """Add embedding column to the database table if it doesn't exist."""
    try:
        # Check if column exists
        try:
            duckdb.sql(f"SELECT embedding FROM {DB_TABLE} WHERE 1=0")
            return True  # Column exists
        except:
            # Add column
            duckdb.sql(f"ALTER TABLE {DB_TABLE} ADD COLUMN embedding DOUBLE PRECISION[];")
            print("Added embedding column to database")
            return True
    except Exception as e:
        print(f"Error adding embedding column: {e}")
        return False

def update_records_with_embeddings(records):
    """Update records with embeddings in the database."""
    if not add_embedding_column():
        return 0
    
    successful_updates = 0
    
    for record_id, embedding in records:
        if not embedding:
            continue
            
        try:
            # Format embedding as array
            embedding_str = f"ARRAY[{', '.join(str(v) for v in embedding)}]"
            
            # Update record
            duckdb.sql(f"UPDATE {DB_TABLE} SET embedding = {embedding_str} WHERE id = '{record_id}';")
            successful_updates += 1
            
            # Print progress periodically
            if successful_updates % 10 == 0:
                print(f"Updated {successful_updates} records")
        except Exception:
            try:
                # Alternative approach using CAST
                values = ', '.join(str(v) for v in embedding)
                duckdb.sql(f"UPDATE {DB_TABLE} SET embedding = CAST('{{{values}}}' AS DOUBLE PRECISION[]) WHERE id = '{record_id}';")
                successful_updates += 1
            except:
                pass
    
    print(f"Updated {successful_updates} records with embeddings")
    return successful_updates

# =====================================================================
# SEARCH SETUP AND INDEXING
# =====================================================================

def setup_search_indexes_and_function():
    """Set up search indexes for text and vector search."""
    # Check prerequisites
    if not add_embedding_column():
        return False
        
    try:
        # Set up text search
        setup_text_search()
        
        # Set up fuzzy search
        setup_fuzzy_search()
        
        # Set up vector search
        setup_vector_index()
        
        # Set up hybrid search function
        setup_hybrid_search_function()
        
        print("Search indexes and functions created successfully")
        return True
    except Exception as e:
        print(f"Error setting up search: {e}")
        return False

def setup_text_search():
    """Set up text search column and GIN index."""
    try:
        # Connect directly to PostgreSQL to create tsvector column and GIN index
        with psycopg2.connect(**PG_PARAMS) as conn:
            with conn.cursor() as cur:
                # Check if fts column exists
                cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{POSTGRES_TABLE}' AND column_name = 'fts'")
                if not cur.fetchone():
                    # Add fts column
                    cur.execute(f"ALTER TABLE {POSTGRES_TABLE} ADD COLUMN fts tsvector")
                    conn.commit()
                    print("Added tsvector column for full-text search")
                
                # Update the tsvector column with weighted text
                cur.execute(f"""
                    UPDATE {POSTGRES_TABLE}
                    SET fts = setweight(to_tsvector('english', coalesce("Title", '')), 'A') || 
                             setweight(to_tsvector('english', coalesce("Location", '')), 'B') ||
                             setweight(to_tsvector('english', coalesce("Details", '')), 'C')
                    WHERE fts IS NULL
                """)
                conn.commit()
                print("Updated tsvector column with weighted document content")
                
                # Create GIN index for full-text search
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_fts_gin ON {POSTGRES_TABLE} USING GIN(fts)")
                conn.commit()
                print("Created GIN index for full-text search")
                
                return True
    except Exception as e:
        print(f"Warning: Text search setup error: {e}")
        return False

def setup_fuzzy_search():
    """Set up trigram indexes for fuzzy matching."""
    try:
        # Connect directly to PostgreSQL
        with psycopg2.connect(**PG_PARAMS) as conn:
            with conn.cursor() as cur:
                # Enable pg_trgm extension if not already enabled
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                conn.commit()
                print("Enabled pg_trgm extension for fuzzy search")
                
                # Create trigram indexes for title and location
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_title_trgm ON {POSTGRES_TABLE} USING GIN (\"Title\" gin_trgm_ops)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_location_trgm ON {POSTGRES_TABLE} USING GIN (\"Location\" gin_trgm_ops)")
                conn.commit()
                print("Created trigram indexes for fuzzy search")
                
                return True
    except Exception as e:
        print(f"Warning: Fuzzy search setup error: {e}")
        return False

def setup_vector_index():
    """Set up HNSW vector index for embeddings using pgvector with direct PostgreSQL connection."""
    try:
        # Connect directly to PostgreSQL
        with psycopg2.connect(**PG_PARAMS) as conn:
            with conn.cursor() as cur:
                try:
                    # Check if vector extension exists
                    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                    if cur.fetchone() is None:
                        print("Warning: vector extension not found in PostgreSQL")
                        return False
                        
                    # Drop existing index if it exists
                    cur.execute(f"DROP INDEX IF EXISTS idx_embedding_hnsw;")
                    
                    # Create HNSW index
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_embedding_hnsw 
                        ON {POSTGRES_TABLE} USING hnsw(embedding vector_ip_ops) 
                        WITH (m=16, ef_construction=64);
                    """)
                    conn.commit()
                    print("Created HNSW index for vector search in PostgreSQL")
                    return True
                except Exception as e:
                    conn.rollback()
                    print(f"Warning: Could not create HNSW index: {e}")
                    
                    # Fallback to basic index
                    try:
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_embedding ON {POSTGRES_TABLE} (embedding);")
                        conn.commit()
                        print("Created basic index for embeddings")
                        return True
                    except Exception as e2:
                        conn.rollback()
                        print(f"Warning: Could not create basic index: {e2}")
                        return False
    except Exception as e:
        print(f"Warning: Vector index setup error: {e}")
        return False

def setup_hybrid_search_function():
    """Set up hybrid search function in PostgreSQL."""
    try:
        # Connect directly to PostgreSQL
        with psycopg2.connect(**PG_PARAMS) as conn:
            with conn.cursor() as cur:
                # Create or replace the hybrid search function
                cur.execute(f"""
                CREATE OR REPLACE FUNCTION hybrid_search(
                  query_text text,
                  query_embedding vector,
                  match_count int,
                  text_weight float DEFAULT 1.0,
                  vector_weight float DEFAULT 1.0,
                  fuzzy_weight float DEFAULT 0.5,
                  rrf_k int DEFAULT 60
                )
                RETURNS TABLE (
                  id uuid,
                  title text,
                  location text,
                  details text,
                  score float,
                  debug jsonb
                )
                LANGUAGE SQL
                AS $$
                WITH text_search AS (
                  SELECT
                    id,
                    "Title",
                    "Location",
                    "Details",
                    ts_rank_cd(fts, websearch_to_tsquery('english', query_text)) as text_score,
                    ROW_NUMBER() OVER(ORDER BY ts_rank_cd(fts, websearch_to_tsquery('english', query_text)) DESC) as rank_text
                  FROM
                    {POSTGRES_TABLE}
                  WHERE
                    websearch_to_tsquery('english', query_text) @@ fts
                  LIMIT match_count * 3
                ),
                fuzzy_search AS (
                  SELECT
                    id,
                    "Title",
                    "Location",
                    "Details",
                    similarity("Title", query_text) as title_sim,
                    similarity("Location", query_text) as location_sim,
                    greatest(similarity("Title", query_text), similarity("Location", query_text)) as fuzzy_score,
                    ROW_NUMBER() OVER(ORDER BY greatest(similarity("Title", query_text), similarity("Location", query_text)) DESC) as rank_fuzzy
                  FROM
                    {POSTGRES_TABLE}
                  WHERE
                    "Title" % query_text OR "Location" % query_text
                  LIMIT match_count * 3
                ),
                vector_search AS (
                  SELECT
                    id,
                    "Title",
                    "Location",
                    "Details",
                    1 - (embedding <#> query_embedding) as vector_score,
                    ROW_NUMBER() OVER(ORDER BY embedding <#> query_embedding ASC) as rank_vector
                  FROM
                    {POSTGRES_TABLE}
                  WHERE
                    embedding IS NOT NULL
                  LIMIT match_count * 3
                )
                SELECT
                  COALESCE(t.id, f.id, v.id) as id,
                  COALESCE(t."Title", f."Title", v."Title") as title,
                  COALESCE(t."Location", f."Location", v."Location") as location,
                  COALESCE(t."Details", f."Details", v."Details") as details,
                  (
                    COALESCE(1.0 / (rrf_k + t.rank_text), 0.0) * text_weight +
                    COALESCE(1.0 / (rrf_k + f.rank_fuzzy), 0.0) * fuzzy_weight +
                    COALESCE(1.0 / (rrf_k + v.rank_vector), 0.0) * vector_weight
                  ) as score,
                  jsonb_build_object(
                    'text', jsonb_build_object('rank', t.rank_text, 'score', t.text_score),
                    'fuzzy', jsonb_build_object('rank', f.rank_fuzzy, 'score', f.fuzzy_score, 'title_sim', f.title_sim, 'location_sim', f.location_sim),
                    'vector', jsonb_build_object('rank', v.rank_vector, 'score', v.vector_score)
                  ) as debug
                FROM
                  text_search t
                  FULL OUTER JOIN fuzzy_search f ON t.id = f.id
                  FULL OUTER JOIN vector_search v ON COALESCE(t.id, f.id) = v.id
                WHERE
                  COALESCE(t.id, f.id, v.id) IS NOT NULL
                ORDER BY
                  score DESC
                LIMIT
                  match_count;
                $$;
                """)
                conn.commit()
                print("Created enhanced hybrid search function with fuzzy matching capability")
                return True
    except Exception as e:
        print(f"Warning: Failed to create hybrid search function: {e}")
        return False

# =====================================================================
# SEARCH FUNCTIONALITY
# =====================================================================

def search(query_text, limit=5):
    """
    Search for items using hybrid search combining text and vector methods.
    
    Args:
        query_text: User's query
        limit: Maximum number of results
        
    Returns:
        tuple: (results, column_names) or None
    """
    try:
        # Generate embedding for the query text
        query_embedding = generate_embedding(query_text)
        if not query_embedding:
            print("Could not generate embedding for query text")
            return None
        
        # Use hybrid search only
        results = hybrid_search(query_text, query_embedding, limit)
        if results:
            return results
                
        # No results found
        print(f"No results found for query: '{query_text}'")
        return None
    except Exception as e:
        print(f"Search error: {e}")
        return None

def hybrid_search(query_text, query_embedding, limit, text_weight=1.0, vector_weight=1.0, fuzzy_weight=0.5):
    """Execute hybrid search using PostgreSQL function."""
    try:
        # Connect directly to PostgreSQL
        with psycopg2.connect(**PG_PARAMS) as conn:
            with conn.cursor() as cur:
                # Format embedding for SQL - using square brackets
                embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
                
                # Call the hybrid search function with explicit casting
                cur.execute(f"""
                    SELECT id, title, location, details, score 
                    FROM hybrid_search(
                        %s, 
                        %s::vector, 
                        %s,
                        %s,
                        %s,
                        %s
                    )
                """, (query_text, embedding_str, limit, text_weight, vector_weight, fuzzy_weight))
                
                # Fetch results
                results = cur.fetchall()
                if results:
                    column_names = ["id", "title", "location", "details", "score"]
                    
                    # Print debug info
                    try:
                        cur.execute(f"""
                            SELECT debug FROM hybrid_search(
                                %s, %s::vector, 1, %s, %s, %s
                            ) LIMIT 1
                        """, (query_text, embedding_str, text_weight, vector_weight, fuzzy_weight))
                        debug_info = cur.fetchone()
                        if debug_info and debug_info[0]:
                            print(f"Search debug info: {debug_info[0]}")
                    except:
                        pass
                        
                    return (results, column_names)
                return None
    except Exception as e:
        print(f"Hybrid search error: {e}")
        return None




