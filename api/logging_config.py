import logging
import json
import time

# Configuración base
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

# Logger principal
logger = logging.getLogger("api")


def log_event(event: str, **kwargs):
    """
    Genera logs estructurados JSON
    """

    payload = {
        "event": event,
        "timestamp": time.time(),
        **kwargs
    }

    logger.info(json.dumps(payload))