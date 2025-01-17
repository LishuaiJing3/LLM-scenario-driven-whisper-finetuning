import logging

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=log_level,
    )
    logging.info("Logging setup complete")
