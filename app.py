import logging
import os
import time
from typing import List

import keyboard
import openai
import psutil
import streamlit as st
from dotenv import load_dotenv
from nemoguardrails import RailsConfig, LLMRails
from openai import OpenAI

# --- Logging Configuration ---

LOG_FILE = "app.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("nemo_playground")

logger.info("=" * 60)
logger.info("Application starting")

# Page config must be the first Streamlit call
st.set_page_config(layout="wide")

# Load environment variables from .env file
load_dotenv()
logger.debug("Environment variables loaded from .env")

# Validate NVIDIA API key
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not NVIDIA_API_KEY:
    logger.error("NVIDIA_API_KEY not found in environment variables")
    st.error("NVIDIA_API_KEY not found. Please set it in your .env file.")
    st.stop()


def mask_api_key(key: str) -> str:
    """Return a masked version of the API key showing only first 4 and last 4 characters."""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"


logger.info(f"NVIDIA_API_KEY loaded: {mask_api_key(NVIDIA_API_KEY)}")

# --- Configuration Loading from Disk ---

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
CONFIG_YML_PATH = os.path.join(CONFIG_DIR, "config.yml")
CONFIG_CO_PATH = os.path.join(CONFIG_DIR, "main.co")


def load_config_from_disk():
    """Load config.yml and main.co content from disk files."""
    try:
        with open(CONFIG_YML_PATH, "r", encoding="utf-8") as f:
            config_yml = f.read()
        logger.debug(f"Loaded config.yml from {CONFIG_YML_PATH} ({len(config_yml)} chars)")
    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_YML_PATH}")
        st.error(f"Configuration file not found: {CONFIG_YML_PATH}")
        st.stop()
    except Exception as e:
        logger.error(f"Failed to read config.yml: {e}", exc_info=True)
        st.error(f"Failed to read config.yml: {e}")
        st.stop()

    try:
        with open(CONFIG_CO_PATH, "r", encoding="utf-8") as f:
            config_co = f.read()
        logger.debug(f"Loaded main.co from {CONFIG_CO_PATH} ({len(config_co)} chars)")
    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_CO_PATH}")
        st.error(f"Configuration file not found: {CONFIG_CO_PATH}")
        st.stop()
    except Exception as e:
        logger.error(f"Failed to read main.co: {e}", exc_info=True)
        st.error(f"Failed to read main.co: {e}")
        st.stop()

    return config_yml, config_co

# --- Session State Initialization (defensive pattern) ---

if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.debug("Session state initialized: messages=[]")

if "guardrails_enabled" not in st.session_state:
    st.session_state.guardrails_enabled = True
    logger.debug("Session state initialized: guardrails_enabled=True")

if "config_yml" not in st.session_state or "config_co" not in st.session_state:
    _config_yml, _config_co = load_config_from_disk()
    if "config_yml" not in st.session_state:
        st.session_state.config_yml = _config_yml
        logger.debug("Session state initialized: config_yml (loaded from disk)")
    if "config_co" not in st.session_state:
        st.session_state.config_co = _config_co
        logger.debug("Session state initialized: config_co (loaded from disk)")


# --- LLMRails Instance Management ---


def validate_kb_folder(config_dir: str) -> dict:
    """Validate the knowledge base folder within the config directory.

    Checks if the kb/ folder exists, counts .md files, and validates UTF-8 encoding.
    Returns status information about the KB folder state.

    Args:
        config_dir: Path to the configuration directory.

    Returns:
        A dict with keys:
            - folder_exists (bool): Whether the kb/ folder exists
            - doc_count (int): Number of valid .md files found
            - skipped_files (list[str]): Filenames that were skipped due to encoding errors
    """
    kb_path = os.path.join(config_dir, "kb")
    result = {"folder_exists": False, "doc_count": 0, "skipped_files": []}

    if not os.path.isdir(kb_path):
        logger.info(f"KB folder not found at '{kb_path}' - RAG is disabled")
        return result

    result["folder_exists"] = True

    md_files = [f for f in os.listdir(kb_path) if f.endswith(".md")]

    if not md_files:
        logger.warning(f"KB folder '{kb_path}' is empty - no documents found, RAG will return no results")
        return result

    valid_count = 0
    for filename in md_files:
        filepath = os.path.join(kb_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                f.read()
            valid_count += 1
        except UnicodeDecodeError:
            logger.error(f"Invalid UTF-8 encoding in KB file: '{filename}' - skipping")
            result["skipped_files"].append(filename)
        except OSError as e:
            logger.error(f"Cannot read KB file: '{filename}' - {e} - skipping")
            result["skipped_files"].append(filename)

    result["doc_count"] = valid_count
    logger.info(f"KB folder validated: {valid_count} document(s) found, {len(result['skipped_files'])} skipped")
    return result


def create_rails_instance() -> LLMRails:
    """Create an LLMRails instance from the config directory.

    Loads configuration from the config/ directory using RailsConfig.from_path(),
    which automatically discovers config.yml, Colang files, and kb/ documents.

    Raises:
        FileNotFoundError: If the config directory does not exist.
        ValueError: If the config directory is invalid or cannot be loaded.
    """
    logger.info(f"Creating new LLMRails instance from config directory: {CONFIG_DIR}")

    # Validate config directory exists
    if not os.path.isdir(CONFIG_DIR):
        error_msg = f"Config directory not found: '{CONFIG_DIR}'"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Validate config directory contains required files
    config_yml_path = os.path.join(CONFIG_DIR, "config.yml")
    if not os.path.isfile(config_yml_path):
        error_msg = f"Config file not found: '{config_yml_path}'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate KB folder and handle edge cases
    kb_status = validate_kb_folder(CONFIG_DIR)

    # Remove invalid UTF-8 files temporarily so the framework doesn't crash on them
    kb_path = os.path.join(CONFIG_DIR, "kb")
    renamed_files = []
    if kb_status["skipped_files"]:
        for bad_file in kb_status["skipped_files"]:
            src = os.path.join(kb_path, bad_file)
            dst = src + ".invalid"
            try:
                os.rename(src, dst)
                renamed_files.append((src, dst))
                logger.debug(f"Temporarily renamed invalid file: '{bad_file}' -> '{bad_file}.invalid'")
            except OSError as e:
                logger.warning(f"Could not rename invalid file '{bad_file}': {e}")

    try:
        config = RailsConfig.from_path(CONFIG_DIR)
    except Exception as e:
        error_msg = f"Failed to load config from '{CONFIG_DIR}': {e}"
        logger.error(error_msg, exc_info=True)
        # Restore renamed files before raising
        for src, dst in renamed_files:
            try:
                os.rename(dst, src)
            except OSError:
                pass
        raise ValueError(error_msg) from e

    # Restore renamed files after successful config load
    for src, dst in renamed_files:
        try:
            os.rename(dst, src)
        except OSError as e:
            logger.warning(f"Could not restore file '{dst}' -> '{src}': {e}")

    init_start = time.time()
    rails = LLMRails(config)
    init_elapsed = time.time() - init_start

    # Log KB pipeline initialization details (Requirement 8.1)
    if hasattr(rails, 'kb') and rails.kb:
        kb = rails.kb
        doc_count = len(config.docs) if config.docs else 0
        chunk_count = len(kb.chunks) if kb.chunks else 0
        logger.info(f"KB initialized: {doc_count} documents loaded, {chunk_count} chunks created")

        if init_elapsed < 1.0:
            logger.info(f"KB index loaded from cache in {init_elapsed:.3f}s")
        else:
            logger.info(f"KB index built in {init_elapsed:.3f}s")

        logger.debug(
            f"Embedding model: {rails.default_embedding_engine}/{rails.default_embedding_model}, "
            f"dimension: {kb.index.embedding_size if kb.index and hasattr(kb.index, 'embedding_size') else 'unknown'}"
        )
    else:
        logger.info("LLMRails instance created without knowledge base")

    logger.info("LLMRails instance created successfully")
    return rails


if "rails" not in st.session_state:
    try:
        st.session_state.rails = create_rails_instance()
    except Exception as e:
        logger.error(f"Failed to create initial LLMRails instance: {e}", exc_info=True)
        st.error(f"Failed to initialize guardrails: {e}")
        st.stop()


# --- Direct Client (Guardrails Disabled) ---

logger.debug("Creating OpenAI direct client for NVIDIA NIM")
direct_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)
logger.info("OpenAI direct client initialized (base_url=https://integrate.api.nvidia.com/v1)")


def generate_direct_response(client: OpenAI, messages: List[dict]) -> str:
    """Generate an unguarded response directly from the model."""
    logger.info(f"Generating direct response (no guardrails), {len(messages)} messages in context")
    logger.debug(f"Last user message: {messages[-1]['content'][:100] if messages else 'N/A'}...")
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-8b-v1",
        messages=messages,
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    )
    response = completion.choices[0].message.content
    logger.info(f"Direct response received: {len(response)} chars")
    logger.debug(f"Direct response preview: {response[:150]}...")
    return response


# --- Message Processing Engines ---

def _extract_kb_results(rails: LLMRails, user_message: str) -> dict:
    """Extract knowledge base retrieval results for the given user message.

    Searches the KB index for relevant chunks matching the user query,
    including similarity scores computed from angular distance.

    Returns a dict with keys:
        - results (list): List of dicts with title, score, and text for each chunk
        - error (str | None): Error type if a failure occurred ("embedding_failure" or "search_failure")
        - error_message (str | None): Human-readable error description
    """
    output = {"results": [], "error": None, "error_message": None}

    if rails.kb is None or rails.kb.index is None:
        logger.debug("No KB or KB index available, returning empty kb_results")
        return output

    import asyncio
    from nemoguardrails.utils import get_or_create_event_loop

    loop = get_or_create_event_loop()
    index = rails.kb.index

    # Step 1: Embed the user query
    try:
        embeddings = loop.run_until_complete(index._get_embeddings([user_message]))
        query_embedding = embeddings[0]
    except Exception as e:
        logger.error(
            f"Embedding generation failed for query '{user_message[:80]}': {e}",
            exc_info=True,
        )
        output["error"] = "embedding_failure"
        output["error_message"] = str(e)
        return output

    # Step 2: Search the Annoy index
    try:
        if hasattr(index, '_index') and index._index is not None:
            raw_results = index._index.get_nns_by_vector(
                query_embedding, 3, include_distances=True
            )
            indices, distances = raw_results

            for idx, distance in zip(indices, distances):
                score = round(1 - distance / 2, 4)
                item = index._items[idx]
                title = item.meta.get("title", "") if item.meta else ""
                body = item.meta.get("body", "") if item.meta else item.text
                output["results"].append({
                    "title": title,
                    "score": score,
                    "text": body[:200] if body else "",
                })
        else:
            # Fallback: use standard search without scores
            results = loop.run_until_complete(
                index.search(user_message, max_results=3)
            )
            for item in results:
                title = item.meta.get("title", "") if item.meta else ""
                body = item.meta.get("body", "") if item.meta else item.text
                output["results"].append({
                    "title": title,
                    "score": None,
                    "text": body[:200] if body else "",
                })

        logger.debug(
            f"KB retrieval returned {len(output['results'])} results for query: "
            f"'{user_message[:80]}'"
        )

    except Exception as e:
        logger.warning(
            f"Annoy index search failed for query '{user_message[:80]}': {e}. "
            "Falling back to response without RAG context."
        )
        output["error"] = "search_failure"
        output["error_message"] = str(e)
        output["results"] = []

    return output


def generate_guarded_response(rails: LLMRails, messages: List[dict]) -> tuple[str, dict]:
    """Generate a response through NeMo Guardrails, returning (response_text, trace).

    NeMo Guardrails handles:
    - Input rails (LLM-based prompt injection / safety check)
    - Intent classification via LLM continuation
    - KB retrieval for grounded responses
    - Off-topic detection via LLM + instructions

    Note: Colang 2.0 only supports passing the latest user message.
    """
    logger.info(f"Generating guarded response, {len(messages)} messages in context")
    logger.debug(f"Last user message: {messages[-1]['content'][:100] if messages else 'N/A'}...")

    # Colang 2.0 only supports user messages - send only the latest user message
    latest_user_messages = [msg for msg in messages if msg["role"] == "user"]
    if latest_user_messages:
        input_messages = [{"role": "user", "content": latest_user_messages[-1]["content"]}]
    else:
        input_messages = messages[-1:]

    logger.debug(f"Sending to rails.generate(): {input_messages}")

    # Let NeMo Guardrails handle everything: input rails, intent classification,
    # KB retrieval, response generation, and output rails.
    try:
        result = rails.generate(messages=input_messages)
        logger.debug(f"rails.generate() returned: {result}")
    except Exception as e:
        logger.error(f"rails.generate() failed: {e}", exc_info=True)
        trace = {
            "engine": "guardrails",
            "status": "error",
            "kb_results": [],
        }
        error_response = (
            "Sorry, I was unable to generate a response due to a system error. "
            "Please try again later."
        )
        return error_response, trace

    # Extract KB retrieval results for the debug trace.
    user_message = input_messages[0]["content"] if input_messages else ""
    kb_output = _extract_kb_results(rails, user_message)
    rag_error = kb_output["error"]
    rag_fallback = rag_error is not None

    # Extract response text defensively (dict-style access)
    if isinstance(result, dict):
        response_text = result.get("content", "")
        # Collect any extra info from the result as trace
        trace = {k: v for k, v in result.items() if k not in ("content", "role")}
    else:
        response_text = getattr(result, "content", str(result))
        trace = {}

    # Classify the response for trace display
    if not response_text or not response_text.strip():
        # Empty response typically means input rails blocked the message
        response_text = "🛡️ Your message was blocked by guardrails. It may have been flagged as unsafe or off-topic."
        trace["blocked"] = True
        trace["engine"] = "guardrails"
        trace["status"] = "input_blocked"
        trace["guardrail_triggered"] = "input_rail"
        logger.info(f"Input rail blocked message: '{input_messages[0]['content'][:80]}...'")
    elif "can't respond to that" in response_text.lower() or "cannot respond" in response_text.lower():
        trace["engine"] = "guardrails"
        trace["status"] = "input_blocked"
        trace["guardrail_triggered"] = "input_rail"
        trace["action"] = "blocked"
    else:
        # Normal response
        if not trace:
            trace = {"engine": "guardrails", "status": "completed", "guardrail_triggered": "none"}

    # Add KB results and RAG fallback metadata to trace
    trace["kb_results"] = kb_output["results"]

    if rag_fallback:
        trace["rag_fallback"] = True
        trace["rag_error"] = kb_output["error"]
        logger.info(
            f"KB retrieval encountered error: {kb_output['error']} "
            "(response generated without additional RAG context in trace)"
        )

    logger.debug(f"KB results in trace: {len(trace['kb_results'])} chunks")
    logger.info(f"Guarded response received: {len(response_text)} chars")
    logger.debug(f"Guarded response: {response_text[:150]}...")
    logger.debug(f"Trace data: {trace}")

    return response_text, trace

# --- Shutdown Handler ---

def shutdown_app():
    """Gracefully terminate the Streamlit application."""
    logger.warning("Shutdown requested by user")
    st.warning("Shutting down...")
    time.sleep(0.5)
    try:
        keyboard.press_and_release("ctrl+w")
        pid = os.getpid()
        logger.info(f"Terminating process PID={pid}")
        p = psutil.Process(pid)
        p.terminate()
    except Exception as e:
        logger.error(f"Shutdown failed, forcing exit: {e}")
        os._exit(0)


# --- Sidebar UI ---

with st.sidebar:
    # Guardrail Toggle
    st.toggle("Guardrails Enabled", value=True, key="guardrails_enabled")

    # Status indicator with color coding
    if st.session_state.guardrails_enabled:
        st.success("🛡️ Guardrails: Enabled")
    else:
        st.warning("⚠️ Guardrails: Disabled")

    st.divider()

    # Config editors
    st.text_area(
        "config.yml",
        height=300,
        key="config_yml",
    )

    st.text_area(
        "main.co",
        height=300,
        key="config_co",
    )

    st.divider()

    # Action buttons
    update_clicked = st.button("Update")
    shutdown_clicked = st.button("⏹️ Shutdown App")

    # Update button handler with hot-reload
    if update_clicked:
        logger.info("Update button clicked - attempting config hot-reload")
        if not st.session_state.config_yml.strip() or not st.session_state.config_co.strip():
            logger.warning("Update aborted: one or both config fields are empty")
            st.warning("Configuration fields cannot be empty. Please provide both config.yml and main.co content.")
        else:
            with st.spinner("Updating guardrails configuration..."):
                try:
                    # Write modified config content to disk before reinitializing
                    with open(CONFIG_YML_PATH, "w", encoding="utf-8") as f:
                        f.write(st.session_state.config_yml)
                    logger.debug(f"Wrote config.yml to {CONFIG_YML_PATH} ({len(st.session_state.config_yml)} chars)")

                    with open(CONFIG_CO_PATH, "w", encoding="utf-8") as f:
                        f.write(st.session_state.config_co)
                    logger.debug(f"Wrote main.co to {CONFIG_CO_PATH} ({len(st.session_state.config_co)} chars)")

                    # Reinitialize from the config directory
                    new_rails = create_rails_instance()
                    st.session_state.rails = new_rails
                    logger.info("Config hot-reload successful - new LLMRails instance active")
                    st.success("Configuration updated!")
                except Exception as e:
                    logger.error(f"Config hot-reload failed: {e}", exc_info=True)
                    st.error(f"Failed to update configuration: {e}")

    # Shutdown button handler
    if shutdown_clicked:
        shutdown_app()

# --- Chat Interface ---

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("trace") is not None:
            with st.expander("🔍 Debug Trace"):
                st.json(message["trace"])

# --- Chat Input and Message Routing ---

if user_message := st.chat_input("Type your message..."):
    logger.info(f"User message received: '{user_message[:100]}...' (guardrails={'enabled' if st.session_state.guardrails_enabled else 'disabled'})")

    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_message)

    # Prepare messages for API (strip trace key)
    messages_for_api = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    logger.debug(f"Messages prepared for API: {len(messages_for_api)} total")

    # Route based on guardrails toggle with error handling
    try:
        if st.session_state.guardrails_enabled:
            response_text, trace = generate_guarded_response(
                st.session_state.rails, messages_for_api
            )
        else:
            response_text = generate_direct_response(direct_client, messages_for_api)
            trace = None

        # Append assistant message to session state only on success
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "trace": trace}
        )
        logger.info(f"Assistant response appended to history (total messages: {len(st.session_state.messages)})")

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response_text)
            if trace is not None:
                with st.expander("🔍 Debug Trace"):
                    st.json(trace)

    except (openai.APIError, openai.APIConnectionError) as e:
        logger.error(f"API Error during message processing: {e}", exc_info=True)
        st.error(f"API Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during message processing: {e}", exc_info=True)
        st.error(f"Error generating response: {e}")
