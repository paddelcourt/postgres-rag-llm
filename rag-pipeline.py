from google import genai
from utils import (
    extract_data_from_source, 
    embed_data, 
    update_records_with_embeddings, 
    setup_search_indexes_and_function,
    search,
    add_embedding_column,
    hybrid_search,
    generate_embedding
)
import duckdb
import os
from dotenv import load_dotenv

# Load environment variables and initialize API client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def connect_to_postgres():
    """Connect to PostgreSQL and display database information"""
    duckdb.sql("ATTACH 'dbname=postgres user=postgres host=127.0.0.1 password=postgres' AS postgres (TYPE postgres);")
    print("Connected to PostgreSQL database")
    
    # Show database statistics
    try:
        count_result = duckdb.sql("SELECT COUNT(*) FROM postgres.real_estate").fetchall()
        print(f"Database has {count_result[0][0]} records in the real_estate table")
        
        if count_result[0][0] > 0:
            sample_data = duckdb.sql("SELECT id, \"Title\", \"Location\" FROM postgres.real_estate LIMIT 3").fetchall()
            print("\nSample records:")
            for record in sample_data:
                print(f"ID: {record[0]}, Title: {record[1]}, Location: {record[2]}")
    except Exception as e:
        print(f"Error checking database: {e}")

def ingest_data(batch_size=10, skip_embeddings=False, skip_updates=False):
    """Process data, generate embeddings, and update the database"""
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
            print("Enhanced search is now enabled with:")
            print("  - Full-text search using PostgreSQL tsvector with GIN indexing")
            print("    (Title fields weighted higher than Location or Details)")
            print("  - Fuzzy matching using PostgreSQL pg_trgm extension")
            print("    (Helps find results with typos or slight variations)")
            print("  - Semantic search using HNSW vector indexing")
            print("  - Results combined with Reciprocal Rank Fusion")
        else:
            print("Search setup encountered errors")

def extract_keywords_from_query(query_text):
    """
    Extract relevant search keywords from a natural language query using Gemini.
    Focuses on main subjects, descriptive terms, attributes, and location references.
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
    """Generate an answer using Gemini based on search results"""
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
If the information doesn't match the question, please say so.

<listings>
{context}
</listings>

Question: {user_text}
"""
    
    try:
        # Call Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return response.text
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"I found matching items but couldn't generate a response: {str(e)[:100]}..."

def interactive_search():
    """Interactive search interface"""
    print("\n=== Enhanced Search Assistant ===")
    print("This system uses a powerful combination of search techniques:")
    print("  • Keyword search - finds exact and related terms")
    print("  • Fuzzy matching - handles typos and variations")
    print("  • Semantic search - understands meaning and context")
    print("\nType your questions or 'quit' to exit")
    
    while True:
        user_input = input("\nYour question: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Thank you for using the Search Assistant. Goodbye!")
            break
        
        print("Processing your query...")
        
        # Extract keywords from user input to enhance search
        enhanced_query = extract_keywords_from_query(user_input)
        
        print("Searching with enhanced keywords...")
        
        # Balance text and vector search weights for best results
        text_weight = 1.0    # Full-text search weight
        vector_weight = 1.2  # Semantic search weight
        fuzzy_weight = 0.7   # Fuzzy search weight
        
        # Generate the embedding
        embedding = generate_embedding(enhanced_query)
        
        # Search using hybrid approach
        if embedding:
            results = hybrid_search(enhanced_query, embedding, 5, text_weight, vector_weight, fuzzy_weight)
        else:
            results = search(enhanced_query)
        
        if not results:
            print("No matching results found.")
            continue
            
        print(f"Found {len(results[0])} relevant results")
            
        # Generate answer based on original user query
        answer = generate_answer(user_input, results)
        
        # Show answer
        print("\nAnswer:", answer)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--batch", type=int, default=10, help="Batch size (default: 10)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-updates", action="store_true", help="Skip database updates")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip data ingestion (search only)")
    
    args = parser.parse_args()
    
    # Connect to database
    print("Connecting to database...")
    connect_to_postgres()
    
    # Set up search indexes
    print("Ensuring search is set up...")
    setup_search_indexes_and_function()
    
    # Run data ingestion if not skipped
    if not args.skip_ingestion:
        print("\n=== Starting Data Processing ===")
        ingest_data(
            batch_size=args.batch,
            skip_embeddings=args.skip_embeddings,
            skip_updates=args.skip_updates
        )
    
    # Start interactive search
    interactive_search()