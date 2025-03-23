from google import genai
from google.genai import types
from utils import (
    extract_data_from_source, 
    embed_data, 
    update_records_with_embeddings, 
    setup_search_indexes_and_function,
    search,
    add_embedding_column
)
import duckdb
import os
from dotenv import load_dotenv

# Connect to database and initialize API client
def connect_to_postgres():
    duckdb.sql("ATTACH 'dbname=postgres user=postgres host=127.0.0.1 password=postgres' AS postgres (TYPE postgres);")
    print("Connected to PostgreSQL database")
    
    # Diagnostic check for data
    try:
        count_result = duckdb.sql("SELECT COUNT(*) FROM postgres.real_estate").fetchall()
        print(f"Database has {count_result[0][0]} records in the real_estate table")
        
        if count_result[0][0] > 0:
            sample_data = duckdb.sql("SELECT id, \"Title\", \"Location\" FROM postgres.real_estate LIMIT 3").fetchall()
            print("\nSample records:")
            for record in sample_data:
                print(f"ID: {record[0]}, Title: {record[1]}, Location: {record[2]}")
    except Exception as e:
        print(f"Diagnostic error: {e}")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ingest_table_embeddings(batch_size=10, skip_embeddings=False, skip_updates=False):
    """
    Process data, generate embeddings, and update the database.
    
    Args:
        batch_size: Process this many records at once
        skip_embeddings: Skip embedding generation for testing
        skip_updates: Skip database updates for testing
    """
    # Make sure embedding column exists
    add_embedding_column()
    
    # Get data from database
    print("Extracting data...")
    data, column_names = extract_data_from_source()
    
    if not data:
        print("No data found")
        return
    
    print(f"Processing {len(data)} records in batches of {batch_size}")
    
    # Find ID column index
    id_index = column_names.index('id')
    
    # Process in batches
    total_records = len(data)
    total_updated = 0
    
    for batch_start in range(0, total_records, batch_size):
        # Get current batch
        batch_end = min(batch_start + batch_size, total_records)
        current_batch = data[batch_start:batch_end]
        
        print(f"\nBatch {batch_start//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")
        
        # Generate embeddings
        if skip_embeddings:
            print("Using dummy embeddings")
            batch_embeddings = [[0.0] * 768 for _ in range(len(current_batch))]
        else:
            print(f"Generating embeddings for {len(current_batch)} records")
            batch_embeddings = embed_data((current_batch, column_names))
        
        # Skip database updates if requested
        if skip_updates:
            print("Skipping database updates")
            continue
        
        # Prepare records for update
        records_to_update = [(row[id_index], embedding) 
                             for row, embedding in zip(current_batch, batch_embeddings)]
        
        # Update database
        print(f"Updating database with embeddings")
        updated = update_records_with_embeddings(records_to_update)
        total_updated += updated
        
        print(f"Progress: {batch_end}/{total_records} records processed")
    
    print(f"\nCompleted: {total_updated}/{total_records} records updated with embeddings")
    
    # Set up search indexes
    if total_updated > 0:
        print("\nSetting up search indexes...")
        if setup_search_indexes_and_function():
            print("Search setup completed successfully")
            print("Hybrid search is enabled with:")
            print("  - Full-text search using PostgreSQL tsvector with GIN indexing")
            print("  - Semantic search using HNSW vector indexing")
            print("  - Results combined with Reciprocal Rank Fusion")
        else:
            print("Search setup encountered errors")

def extract_keywords_from_query(query_text):
    """
    Use Gemini to extract relevant search keywords from a natural language query.
    This makes search more effective by focusing on the most important terms.
    
    Args:
        query_text: The user's natural language query
        
    Returns:
        str: Extracted keywords suitable for search
    """
    try:
        prompt = f"""
        Extract the most relevant search keywords from this query: "{query_text}"
        
        Focus on extracting:
        - Main subjects/objects
        - Important descriptive terms
        - Specific attributes or qualifiers
        - Numbers and measurements
        - Location references
        
        Return ONLY the essential search keywords separated by spaces, with no explanations.
        Example input: "I'm looking for something with 3 bedrooms in a nice area"
        Example output: "3 bedrooms nice area"
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        enhanced_query = response.text.strip()
        print(f"Enhanced search query: '{enhanced_query}'")
        return enhanced_query
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return query_text  # Return original query if keyword extraction fails

def generate_answer(user_text, search_results):
    """Generate an answer using Gemini based on search results."""
    if not search_results:
        return "I couldn't find any relevant information for your query."
    
    results_data, column_names = search_results
    
    # Format search results as context
    context = ""
    for row in results_data:
        context += f"Title: {row[1]}\n"
        context += f"Location: {row[2]}\n"
        context += f"Details: {row[3]}\n\n"
    
    # Create prompt
    prompt = f"""
Here are some relevant listings. Use them to answer my question.
Always refer to the information in the listings when answering.
If the information is not matching with the question, just say so. 
I.e the rooms are not matching with the number of bedrooms in the question.

<listings>
{context}
</listings>

Question: {user_text}
"""
    
    try:
        # Call Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction="You're an expert at answering questions based on the provided listings. Always use the documents provided in the context."
            ),
            contents=prompt
        )
        
        return response.text
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"I found matching items but couldn't generate a response: {str(e)[:100]}..."

def interactive_search():
    """Interactive search loop."""
    print("Welcome to the Advanced Search Assistant!")
    print("This system uses hybrid search with:")
    print("  - PostgreSQL full-text search (tsvector/tsquery with GIN index)")
    print("  - HNSW vector indexing for semantic search")
    print("  - Reciprocal Rank Fusion to combine results")
    print("\nAsk questions about the data or type 'quit' to exit")
    
    while True:
        user_input = input("\nYour question: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Thank you for using the Advanced Search Assistant. Goodbye!")
            break
        
        print("Processing your query...")
        
        # Extract keywords from user input to enhance search
        enhanced_query = extract_keywords_from_query(user_input)
        
        print("Searching with hybrid approach (text + vector)...")
        
        # Search for relevant items using the enhanced query
        results = search(enhanced_query)
        
        if not results:
            print("No matching results found for your query.")
            continue
            
        print(f"Found {len(results[0])} relevant results")
            
        # Generate answer based on original user query
        answer = generate_answer(user_input, results)
        
        # Show answer
        print("\nAnswer:", answer)

if __name__ == "__main__":
    import sys
    
    # Default settings
    batch_size = 10
    skip_embeddings = False
    skip_updates = False
    skip_ingestion = False
    
    # Process command line arguments
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--batch="):
                try:
                    batch_size = int(arg.split("=")[1])
                except:
                    print(f"Invalid batch size: {arg}")
            elif arg == "--skip-embeddings":
                skip_embeddings = True
                print("Will skip embedding generation")
            elif arg == "--skip-updates":
                skip_updates = True
                print("Will skip database updates")
            elif arg == "--skip-ingestion":
                skip_ingestion = True
                print("Will skip data ingestion")
            elif arg == "--help":
                print("\nRAG Pipeline Options:")
                print("  --batch=N           Set batch size (default: 10)")
                print("  --skip-embeddings   Skip embedding generation")
                print("  --skip-updates      Skip database updates")
                print("  --skip-ingestion    Skip data ingestion (search only)")
                print("  --help              Show this help message")
                sys.exit(0)
    
    # Connect to database
    print("Connecting to database...")
    connect_to_postgres()
    
    # Ensure search indexes and functions are set up
    print("Ensuring search setup is complete...")
    setup_search_indexes_and_function()
    
    # Run data ingestion if not skipped
    if not skip_ingestion:
        print("\n=== Starting Data Processing ===")
        ingest_table_embeddings(
            batch_size=batch_size,
            skip_embeddings=skip_embeddings,
            skip_updates=skip_updates
        )
    
    # Start interactive search
    print("\n=== Starting Interactive Search ===")
    interactive_search()