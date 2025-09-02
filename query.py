import sys
import logging
from argparse import ArgumentParser
from pipeline import query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_result(r: dict) -> str:
    """Format a single search result with page number and summary."""
    score_stars = "★" * int(r['score'] * 5)
    return (
        f"\n[{score_stars}] Score: {r['score']:.3f}\n"
        f"📄 Document: {r['doc_id']} (Page {r.get('page', 1)})\n"
        f"📝 Summary: {r['summary']}\n"
        f"{'-' * 80}"
    )

def main():
    parser = ArgumentParser(description='Query the RAG system')
    parser.add_argument('text', type=str, help='Query text')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    args = parser.parse_args()

    try:
        response = query(args.text, args.top_k)
        
        print(f"\n🔍 Search Results for: '{args.text}'")
        print("=" * 80)
        
        # Print individual results
        for r in response["results"]:
            print(format_result(r))
            
        # Print final answer
        print("\n📚 Final Answer:")
        print("=" * 80)
        print(response["answer"])
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
