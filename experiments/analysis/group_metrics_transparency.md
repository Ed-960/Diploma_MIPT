# Group (заказ на несколько человек): прозрачность метрик

Отчёт для защиты: откуда **52.83% / 62.26% (+9.43)** и что было исправлено offline без перегенерации диалогов.

## Три уровня оценки (paired N=159)

| Метрика                                                   | no-rag | RAG    | Δ (RAG − no-rag) |
| --------------------------------------------------------- | ------ | ------ | ---------------- |
| **Только judge** (`judge.parsed.success_at_1`)            | 14.47% | 23.90% | **+9.43**        |
| **После offline rescore** (`metrics.success_at_1` в JSON) | 52.83% | 62.26% | **+9.43**        |
| Парные победы (только RAG / только no-rag)                | 28     | 43     | +15 вопросов     |

**Вывод:** разрыв **+9.4 п.п. совпадает** с чистым judge. Rescore поднял **обе** стороны примерно на +38 п.п. (добавил 61 диалог с `group_completeness≥1` без глобального CV), **не сдвинул** разницу между RAG и no-rag.

## Что было неправильно в сырых данных (исправлено честно)

1. **`expected_item: null`** в JSON при прогоне — банк уже содержал целевые блюда; восстановлено из `questions/group_questions.json`.
2. **Ложный CV «Milk contains allergen milk»** — в меню есть позиция «Milk», фраза про аллергию попадала в `mentioned_items`.
3. **CV по ограничениям с `"for": "child6"`** применялся ко всем упомянутым блюдам — для group неверно; такие ограничения оставлены judge, эвристика их не штрафует глобально.

## Что делает offline rescore (не LLM)

Условие **pass** в `metrics.success_at_1` для group:

- judge уже поставил `success_at_1`, **или**
- `group_completeness ≥ 1.0` **и** нет глобального `constraint_violation` (эвристика после фиксов) **и** нет `hallucination` по judge.

В каждом `question_*.json`: поле `metrics_rescore_note`, `metric_sources.success_at_1 = group_offline_rescore`.

## Ограничение эксперимента (важно сказать на защите)

- **no-rag** (`retrieval_mode=none`): в промпт добавляется `extra_grounding_context` с `constraint_fit_candidates` (до 8 подходящих блюд из банка).
- **RAG** (`retrieval_mode=vector`): только Chroma + rewrite, без этого блока.

Сравнение group **не полностью симметрично** по дизайну прогона. Для строгой симметрии нужен no-rag с `--disable_question_grounding` (новый прогон).

## Что показывать на слайде

1. **Основная строка для диплома (консервативно):** judge-only **14.5% → 23.9% (+9.4)**.
2. **Дополнительно:** после исправления багов CV и критерия «полнота группы» — **52.8% → 62.3% (+9.4)** (тот же Δ).
3. **Не использовать** для group одну только `success@1` как «попадание в Side Salad» — для multi-person это нерелевантно.

Обновление: `make repair-group-rescore` + `make analyze-question-experiments`.

## Пересудить judge без перегенерации диалогов

```bash
make rejudge-group
make analyze-question-experiments
```

Скрипт `scripts/rejudge_group_experiments.py`: только `DialogJudge` по сохранённым `client`/`cashier` репликам, `expected_item` из банка, `metrics = build_metrics_from_judge` (без `group_offline_rescore`). На 159×2 файлов — много вызовов LLM; с Ollama может идти долго, быстрее через API (`make rejudge-group` с `JUDGE_MODEL` / `question-experiment-*-api`).
