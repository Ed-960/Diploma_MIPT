# Индекс отчётов (`make reports` / `run_all_reports.py`)

Источник: incremental + rows.json.

## Question-эксперименты (`experiments/analysis/`)

| Файл | Назначение |
|------|------------|
| `summary_by_category.json` | Метрики по категориям + `pooled_all_categories` |
| `paired_mcnemar.json` | McNemar по категориям и пулу |
| `summary_table.md` | Таблица для презентации / диплома (Markdown) |
| `summary_table.html` | Та же сводка — **кратко для слайдов** (понятные подписи; техника в «Как считалось») |
| `conclusions_ru.md` | Авто-выводы по категориям (с именами метрик в коде) |
| `results_summary_ru.md` | **Итоги и выводы одним текстом** — для слайда / заключения |
| `charts/success_at_1_by_category.png` | Столбцы success@1 |
| `charts/success_at_1_complement_by_category.png` | 100 − success@1 (доля без цели @1) |
| `charts/hallucination_by_category.png` | Столбцы hallucination |
| `charts/rag_retrieval_hit_by_category.png` | RAG retrieval @k |
| `charts/pooled_overall.png` | Пул всех категорий |

Обновление: перезапустите `make reports`.
