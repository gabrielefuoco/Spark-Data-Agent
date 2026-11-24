import streamlit as st
import os
import logging
import threading
import asyncio
import warnings

# Importazioni LlamaIndex
try:
    from llama_index.core import Settings
    from llama_index.llms.gemini import Gemini
    from llama_index.core.agent.workflow import AgentWorkflow
except ImportError as e:
    st.error(f"Errore importazione LlamaIndex: {e}")
    st.stop()

from tools import create_tools

# Configurazione logging
logger = logging.getLogger(__name__)

# Sopprime warnings specifici
warnings.filterwarnings("ignore", message=".*Call to deprecated class Gemini.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# --- HELPER PER ESECUZIONE ASINCRONA IN STREAMLIT ---
@st.cache_resource
def get_background_event_loop():
    loop = asyncio.new_event_loop()
    def _start_loop(loop):
        try:
            asyncio.set_event_loop(loop)
            loop.run_forever()
        except Exception as e:
            logger.error(f"Errore nel loop di background: {e}")
    t = threading.Thread(target=_start_loop, args=(loop,), daemon=True, name="AgentWorkflow_Loop")
    t.start()
    return loop

async def _run_workflow_and_collect(agent_workflow, user_msg: str, memory) -> str:
    maybe_handler = agent_workflow.run(user_msg=user_msg, memory=memory)
    if asyncio.iscoroutine(maybe_handler):
        handler = await maybe_handler
    else:
        handler = maybe_handler

    chunks = []
    try:
        stream_iter = getattr(handler, "stream_events", None)
        if stream_iter is not None:
            async for ev in handler.stream_events():
                if getattr(ev, "delta", None):
                    chunks.append(str(ev.delta))
                elif getattr(ev, "tool_name", None):
                    out = getattr(ev, "tool_output", None)
                    chunks.append(f"[Tool:{ev.tool_name}] {out}")
        final = None
        if asyncio.iscoroutine(handler) or hasattr(handler, "__await__"):
            try:
                final = await handler
            except Exception:
                final = None
        else:
            final = getattr(handler, "response", None) or getattr(handler, "result", None)
    except Exception as e:
        chunks.append(f"[STREAM ERROR] {e}")
        final = None

    if final is not None:
        try:
            return str(final)
        except Exception:
            pass
    return "".join(chunks) or "(Nessun output generato dal workflow)"

# --- CONFIGURAZIONE AGENTE ---
@st.cache_resource
def setup_agent_workflow(system_prompt: str, temperature=0.7):
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            try:
                with open("src/api.env") as f:
                    for line in f:
                        if line.startswith("GEMINI_API_KEY="):
                            gemini_api_key = line.strip().split("=", 1)[1]
                            break
            except FileNotFoundError:
                pass

        if not gemini_api_key:
            st.error("GEMINI_API_KEY non trovata")
            st.stop()

        llm = Gemini(api_key=gemini_api_key, model="models/gemini-2.5-flash-lite", temperature=temperature)
        Settings.llm = llm

        tools = create_tools()
        workflow = AgentWorkflow.from_tools_or_functions(
            tools,
            llm=llm,
            system_prompt=system_prompt,
        )
        return workflow

    except Exception as e:
        st.error(f"Errore nella configurazione di AgentWorkflow: {e}")
        return None
