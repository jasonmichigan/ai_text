"""ai_text_files — package for the ai_text notebook.

  • Output Quality system removed entirely (no profiles, no validation pass).
    The model writes whatever length your prompt requests. Say "write a
    2000-word essay" or "create a 3-page brief" and the model targets that.
  • PDF support added. Read PDFs as input context AND generate PDFs as
    output. Triggered by "create a pdf about ...".

Public conveniences:
    config          all runtime globals; mutate via attribute access
    logger          log_event, log_write, log_section, log_exception,
                    install_log_wrapper, install_log_handler_wrapper

Submodules are NOT auto-imported here; each has side effects (folder
creation, Ollama probing, print output). The notebook imports them
in a deliberate order — see ai_text.ipynb cell 2.
"""

__version__ = "2.0.0"
__notebook_version__ = "ai_text"
