import logging

logging.basicConfig(
    format="[%(filename)s:%(funcName)s] [%(levelname)s] %(message)s", level=logging.INFO,
    handlers=[logging.FileHandler("log.log"), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)