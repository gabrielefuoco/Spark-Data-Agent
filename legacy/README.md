# legacy/ — the pre-rewrite implementation

This is the original academic code, kept **runnable and untouched** as the
reference for the rewrite. It is not the future of the project and it is not
imported by anything under `src/`.

- `Agente.py`, `agent_handler.py`, `tools.py` — LlamaIndex `AgentWorkflow` over Gemini
- `data_manager.py`, `spark_utils.py` — the flat Parquet pipeline and the analysis tools
- `preprocess.py` — raw XML/TSV to Parquet
- `pages/Analisi_Manuale.py` — the Streamlit page
- `requirements.txt` — the old dependency set (Python 3.12, PySpark 3.4.4, LlamaIndex)

It still contains the defects the revamp exists to remove: four blocking
`.count()` calls at startup, `groupBy(...).agg(collect_list("text"))`, wildcard
`pyspark.sql.functions` imports that shadow `min`, `max` and `sum`,
unpartitioned writes. Those defects are the *evidence* for
`.scratch/spark-data-agent-revamp/map.md`, which is why the code is preserved
rather than deleted.

Delete this directory when ticket 06 lands.
