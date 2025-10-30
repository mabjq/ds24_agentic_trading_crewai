import logging
import signal
import sys
from typing import Dict, Any

from crewai.tools import tool

logger = logging.getLogger(__name__)


@tool("Simulate human input for feedback during approval. Call this tool to prompt the user and get their response.")
def human_input_tool(data: Dict[str, Any] = {}) -> str:
    """Tool to get user feedback via input() simulation. Returns the user's string response.

    Args:
        data: Dict with optional 'kwargs' for prompt override.

    Returns:
        str: User's input feedback.
    """
    kwargs = data.get("kwargs", {})
    prompt = kwargs.get(
        "prompt",
        "Approve or comment on recommendations? (Enter to approve or provide feedback): ",
    )
    logger.info(f"Human input tool called with prompt: {prompt}")
    try:
        # Timeout handler for input (30 sec fallback)
        def timeout_handler(signum: int, frame: object) -> None:
            raise TimeoutError("Input timeout after 30 seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        print("\n" * 5 + prompt, end="", flush=True)
        sys.stdout.flush()
        user_response = input()
        signal.alarm(0)

        logger.info(f"User responded: {user_response[:50]}...")
        return user_response.strip() if user_response else "APPROVED"
    except TimeoutError:
        logger.warning("Input timed out after 30 seconds; using fallback 'APPROVED'")
        return "APPROVED"
    except Exception as e:
        logger.error(f"Input error: {e}")
        return "APPROVED"