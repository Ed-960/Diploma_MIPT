set OLLAMA_NUM_PARALLEL=8 && ollama serve

make dataset-rag-from-profiles NUM=20 PROFILES_FILE=profiles_1000.json OUT_RAG=dialogs_rag PRINT_TRACE=1 TRACE_VERBOSE=1 WORKERS=8