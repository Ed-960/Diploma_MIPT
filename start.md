set OLLAMA_NUM_PARALLEL=8 && ollama serve

# Vector RAG (Chroma)
make dataset-rag-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_RAG=dialogs_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8
make dataset-rag-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_RAG=dialogs_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8 SHUFFLE_PROFILES=1 REALISTIC_CASHIER=1

# Graph RAG (та же схема переменных, каталог по умолчанию OUT_GRAPH_RAG=dialogs_graph_rag)
make dataset-rag-graph-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_GRAPH_RAG=dialogs_graph_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8
make dataset-rag-graph-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_GRAPH_RAG=dialogs_graph_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8 SHUFFLE_PROFILES=1 REALISTIC_CASHIER=1

# Визуал графа меню (graph-RAG): Mermaid + HTML, затем открыть docs/menu_graph_rag.html в браузере
make visualize-menu-graph
make visualize-menu-graph-open
