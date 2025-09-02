import logging
from pipeline import rag_flow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting RAG pipeline...")
    try:
        metrics = rag_flow()
        logger.info(f"Pipeline completed successfully.")
        logger.info(f"Metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
