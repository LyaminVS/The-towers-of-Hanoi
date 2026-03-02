# Tower of Hanoi — Reinforcement Learning

Проект обучения RL-агента игре «Ханойская башня».

## Правила

- Все диски изначально на первой палке (меньшие на больших).
- Цель: перенести все диски на третью палку за минимальное число ходов.
- Правило: **нельзя класть диск на диск меньшего размера**.

## State Space

`[палка каждого диска, высота каждого диска]`

## Action Space

Переместить один из **верхних** дисков на другую палку: `(from_stick, to_stick)`.

## Rewards

| Событие | Награда |
|---------|---------|
| Каждый шаг | -1 |
| Недопустимый ход (диск на меньший) | -10 |
| Правильное размещение на 3-й палке | +10 (опционально) |
| Нарушение правила (death) | большой штраф, эпизод завершается |

## Распределение работы

См. [TEAM.md](TEAM.md) — распределение по 5 человекам (кто какой файл пишет).

## Структура проекта

```
RL_proj2/
├── config/              # Конфигурация
│   ├── __init__.py
│   └── settings.py
├── env/                 # Среда Tower of Hanoi
│   ├── __init__.py
│   ├── environment.py
│   ├── state.py
│   ├── actions.py
│   └── rewards.py
├── agent/               # RL-агенты (policy gradient)
│   ├── __init__.py
│   ├── base_agent.py   # Базовый класс
│   ├── policy.py       # Нейросети: PolicyNetwork, ValueNetwork
│   ├── reinforce.py   # REINFORCE
│   ├── reinforce_baseline.py  # REINFORCE + baseline
│   └── trpo.py         # TRPO
├── training/            # Обучение
│   ├── __init__.py
│   ├── trainer.py
│   └── logger.py       # Логирование в файл и консоль
├── utils/               # Утилиты
│   ├── __init__.py
│   └── params.py       # save_eval_params, load_eval_params
├── run/                 # Скрипты запуска
│   ├── __init__.py
│   ├── train.py         # python run/train.py
│   ├── evaluate.py      # python run/evaluate.py
│   └── play.py          # python run/play.py — игра вручную
├── requirements.txt
└── README.md
```

## Запуск

```bash
pip install -r requirements.txt

# Обучение
python run/train.py --num_disks 3 --num_episodes 1000

# Оценка обученной модели
python run/evaluate.py --load_model path/to/model --render

# Игра вручную
python run/play.py --num_disks 3
```
