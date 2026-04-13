set OLLAMA_NUM_PARALLEL=8 && ollama serve

# Vector RAG (Chroma)
make dataset-rag-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_RAG=dialogs_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8
make dataset-rag-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_RAG=dialogs_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8 SHUFFLE_PROFILES=1 REALISTIC_CASHIER=1

# Graph RAG (та же схема переменных, каталог по умолчанию OUT_GRAPH_RAG=dialogs_graph_rag)
make dataset-rag-graph-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_GRAPH_RAG=dialogs_graph_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8
make dataset-rag-graph-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_GRAPH_RAG=dialogs_graph_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8 SHUFFLE_PROFILES=1 REALISTIC_CASHIER=1

# Визуал графа меню (graph-RAG) -> docs/menu_graph_rag.mmd, menu_graph_rag_data.json, menu_graph_rag.html
make visualize-menu-graph
make visualize-menu-graph-open

# Полный граф (все рёбра по min-weight, без лимита 380)
make visualize-menu-graph-full
make visualize-menu-graph-full-open
# Подграф вокруг блюд: переменные FOCUS, FOCUS_HOPS, FOCUS_STYLE, MENU_GRAPH_MAX_EDGES (см. make help)
make visualize-menu-graph-focus
make visualize-menu-graph-focus-open
# Примеры фокуса (имена как в меню JSON)
make visualize-menu-graph-focus FOCUS="Big Mac"
make visualize-menu-graph-focus-open FOCUS="McChicken,Apple Pie"
make visualize-menu-graph-focus FOCUS="Big Mac,Quarter Pounder with Cheese" FOCUS_HOPS=0
make visualize-menu-graph-focus FOCUS="Big Mac" FOCUS_STYLE=induced
make visualize-menu-graph-focus-open FOCUS="Big Mac" MENU_GRAPH_MAX_EDGES=40

make voice-browser-api
make voice-browser-ollama