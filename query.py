import sys
import logging
from argparse import ArgumentParser
from pipeline import query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_result(r: dict, max_text_len: int = 300) -> str:
    """Format a single search result with highlighting and context."""
    score_stars = "★" * int(r['score'] * 5)  # Convert score to star rating
    text = r.get('text', '[Text not found]')
    preview = text[:max_text_len] + "..." if len(text) > max_text_len else text
    
    return (
        f"\n[{score_stars}] Score: {r['score']:.3f}\n"
        f"📄 Document: {r['doc_id']}\n"
        f"📍 Chunk: {r['chunk_id']}\n"
        f"📝 Content:\n{preview}\n"
        f"{'-' * 80}"
    )

def main():
    parser = ArgumentParser(description='Query the RAG system')
    parser.add_argument('text', type=str, help='Query text')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--max-length', type=int, default=300, help='Max preview length')
    args = parser.parse_args()

    try:
        results = query(args.text, args.top_k)
        
        print(f"\n🔍 Search Results for: '{args.text}'")
        print("=" * 80)
        
        for r in results:
            print(format_result(r, max_text_len=args.max_length))
            
        # Show combined context
        print("\n📚 Combined Context:")
        print("=" * 80)
        context = "\n\n".join(r['text'] for r in results)
        print(context)
            
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
