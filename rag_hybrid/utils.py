import logging

def get_logger(name: str):
    """
    Returns a logger with the specified name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def normalize_text(text: str) -> str:
    """
    Normalizes whitespace in text.
    """
    import re
    return re.sub(r'\s+', ' ', text).strip()
