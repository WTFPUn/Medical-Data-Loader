import logging

logging.basicConfig(
    format="[%(levelname)s][%(name)s][%(contexts)s]: %(message)s ", level=logging.INFO,
    handlers=[logging.FileHandler("log.log"), logging.StreamHandler()]
)

logger = logging.getLogger("main")