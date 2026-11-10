import logging
from src import config

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

_client: Langfuse | None = None
_handler: CallbackHandler | None = None


def get_langfuse_client() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_BASE_URL,
        )
        logger.info("Langfuse client initialized (host=%s)", config.LANGFUSE_BASE_URL)
    return _client


def get_callback_handler() -> CallbackHandler:
    global _handler
    if _handler is None:
        _handler = CallbackHandler()
    return _handler


def resolve_trace_id(handler) -> str | None:
    """v3 exposes last_trace_id; v2 exposes get_trace_id()."""
    tid = getattr(handler, "last_trace_id", None)
    if tid:
        return tid
    if hasattr(handler, "get_trace_id"):
        try:
            return handler.get_trace_id()
        except Exception:
            return None
    return None


def score_trace(trace_id: str, name: str, value: float, comment: str = "") -> None:
    client = get_langfuse_client()
    client.create_score(
        trace_id=trace_id,
        name=name,
        value=value,
        comment=comment or None,)