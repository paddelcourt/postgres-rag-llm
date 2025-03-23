"""
PostgreSQL RAG Pipeline with HNSW Vector Search and Reciprocal Rank Fusion
--------------------------------------------------------------------------
This script demonstrates a RAG pipeline with PostgreSQL using:
- HNSW vector indexing for semantic search
- GIN indexing for full-text search
- Reciprocal Rank Fusion for combining results
"""

from google import genai
import os
from dotenv import load_dotenv
import argparse
import psycopg2
from utils import (
    get_connection,
    setup_database,
    update_embeddings,
    hybrid_search,
    get_all_data
)

# Load environment variables and initialize API client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def connect_to_postgres():
    """Connect to PostgreSQL and display database information"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Show database statistics
                cur.execute("SELECT COUNT(*) FROM real_estate")
                count = cur.fetchone()[0]
                print(f"Database has {count} records in the real_estate table")
                
                if count > 0:
                    cur.execute("SELECT id, \"Title\", \"Location\" FROM real_estate LIMIT 3")
                    sample_data = cur.fetchall()
                    print("\nSample records:")
                    for record in sample_data:
                        print(f"ID: {record[0]}, Title: {record[1]}, Location: {record[2]}")
    except Exception as e:
        print(f"Error checking database: {e}")

def ingest_data(batch_size=10, skip_embeddings=False, skip_updates=False):
    """Process data, generate embeddings, and update the database"""
    # Get data from database
    print("Extracting data...")
    data, column_names = get_all_data()
    
    if not data:
        print("No data found")
        return
    
    print(f"Processing {len(data)} records in batches of {batch_size}")
    
    # Process in batches
    total_records = len(data)
    total_updated = 0
    
    for batch_start in range(0, total_records, batch_size):
        # Get current batch
        batch_end = min(batch_start + batch_size, total_records)
        current_batch = data[batch_start:batch_end]
        
        print(f"\nBatch {batch_start//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")
        
        # Skip updates if requested
        if skip_updates:
            print("Skipping database updates")
            continue
        
        # Update embeddings
        batch_data = [row for row in current_batch]
        updated = update_embeddings(batch_data, column_names)
        total_updated += updated
        
        print(f"Progress: {batch_end}/{total_records} records processed")
    
    print(f"\nCompleted: {total_updated}/{total_records} records updated with embeddings")
    
    # Ensure search indexes are properly set up
    print("\nSetting up search indexes...")
    setup_database()
    print("\nSearch setup complete! The system now supports:")
    print("  • Full-text search: Finds keyword matches (Title words count more)")
    print("  • Fuzzy matching: Handles typos and variations")
    print("  • Semantic search: Understands meaning using AI embeddings")
    print("  • Results combined with Reciprocal Rank Fusion (RRF)")

def extract_keywords_from_query(query_text):
    """Extract key search terms from a natural language query using Gemini."""
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
    
    # Create prompt for Gemini
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
        # Generate response
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
    print("  • Results combined with Reciprocal Rank Fusion")
    print("\nType your questions or 'quit' to exit")
    
    while True:
        # Get user input
        user_input = input("\nYour question: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Thank you for using the Search Assistant. Goodbye!")
            break
        
        # Process query and search
        print("Processing your query...")
        enhanced_query = extract_keywords_from_query(user_input)
        
        print("Searching with enhanced keywords...")
        
        # Balance search weights for best results
        text_weight = 1.0    # Full-text search weight
        vector_weight = 1.2  # Semantic search weight (slightly higher)
        
        # Perform hybrid search
        results = hybrid_search(enhanced_query, 5, text_weight, vector_weight)
        
        # Generate and display answer
        if not results:
            print("No matching results found.")
            continue
            
        print(f"Found {len(results[0])} relevant results")
        answer = generate_answer(user_input, results)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG Pipeline with PostgreSQL")
    parser.add_argument("--batch", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-updates", action="store_true", help="Skip database updates")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip data ingestion (search only)")
    
    args = parser.parse_args()
    
    # Connect to database
    print("Connecting to database...")
    connect_to_postgres()
    
    # Set up search indexes
    print("Ensuring search is set up...")
    setup_database()
    
    # Run data ingestion if not skipped
    if not args.skip_ingestion:
        print("\n=== Starting Data Processing ===")
        ingest_data(
            batch_size=args.batch,
            skip_updates=args.skip_updates
        )
    
    # Start interactive search
    interactive_search()