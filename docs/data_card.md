# Данные

## GSM8K (train / test)
- Источник: Hugging Face `openai/gsm8k` (config `main`)
- Лицензия: TODO (проверить на странице датасета)
- Как получены: TODO (`datasets.load_dataset`, дата загрузки, версия)
- Объекты: TODO (число задач, поля), `python -m src.prepare_data` пишет статистику в `results/`
- Проблемы: TODO (аннотации калькулятора `<<...>>`, длины, дубликаты)

## MetaMathQA
- Источник: `meta-math/MetaMathQA`
- Лицензия: TODO
- Особенность: получен из тренировочных частей GSM8K и MATH, формат ответа приведён к `#### N`
- Проверка утечек: фильтр по общим 10-граммам с GSM8K test, число удалённых строк в `results/data_report_*.json`

## Бенчмарки для оценки
GSM8K test (exact match по итоговому числу), ARC-Easy/Challenge, HellaSwag, MMLU (подвыборка).
Для каждого: источник, лицензия, число объектов, как извлекается ответ — TODO.
