# Распределение работы по проекту (5 человек)

## Волны параллельной разработки

Все задачи внутри одной волны выполняются **параллельно**. Следующая волна стартует после завершения предыдущей.

---

## Волна 1 — независимые модули (все стартуют одновременно)

| Человек | Файлы | Зависимости |
|---------|-------|-------------|
| **Person 1** | `env/state.py` | нет |
| **Person 2** | `env/actions.py` | нет |
| **Person 3** | `env/rewards.py` | нет |
| **Person 4** | `agent/base_agent.py`, `agent/policy.py` | нет |
| **Person 5** | `config/settings.py`, `utils/params.py`, `training/logger.py` | нет |

**Перед стартом:** согласовать формат state `(sticks, heights)` и action `(from_stick, to_stick)` — см. описания в файлах.

---

## Волна 2 — после волны 1

| Человек | Файлы | Зависимости |
|---------|-------|-------------|
| **Person 1** | `env/environment.py`, `env/__init__.py` | state, actions, rewards |
| **Person 2** | `agent/reinforce.py` | base_agent, policy |
| **Person 3** | `agent/reinforce_baseline.py` | base_agent, policy |
| **Person 4** | `agent/trpo.py` | base_agent, policy |
| **Person 5** | `agent/__init__.py` (create_agent) | base_agent, reinforce, reinforce_baseline, trpo |

---

## Волна 3 — после волны 2

| Человек | Файлы | Зависимости |
|---------|-------|-------------|
| **Person 1** | интеграция, тесты env | — |
| **Person 2** | `training/trainer.py`, `training/__init__.py` | env, agent |
| **Person 3** | `run/train.py` | env, agent, training, config |
| **Person 4** | `run/evaluate.py` | env, agent, training, utils |
| **Person 5** | `run/play.py`, `utils/__init__.py`, `run/__init__.py` | env (play не использует agent) |

---

## Детализация по файлам

### Person 1
- **Волна 1:** `env/state.py` — get_initial_state, state_to_observation, observation_to_state, is_terminal_state, get_state_hash
- **Волна 2:** `env/environment.py` — TowerOfHanoiEnv, create_env; `env/__init__.py`

### Person 2
- **Волна 1:** `env/actions.py` — get_action_space, get_valid_actions, action_to_index, index_to_action, is_valid_move
- **Волна 2:** `agent/reinforce.py` — REINFORCEAgent
- **Волна 3:** `training/trainer.py`, `training/__init__.py`

### Person 3
- **Волна 1:** `env/rewards.py` — класс Reward
- **Волна 2:** `agent/reinforce_baseline.py` — REINFORCEBaselineAgent
- **Волна 3:** `run/train.py`

### Person 4
- **Волна 1:** `agent/base_agent.py`, `agent/policy.py` — BaseAgent, PolicyNetwork, ValueNetwork
- **Волна 2:** `agent/trpo.py` — TRPOAgent
- **Волна 3:** `run/evaluate.py`

### Person 5
- **Волна 1:** `config/settings.py`, `utils/params.py`, `training/logger.py`
- **Волна 2:** `agent/__init__.py` — create_agent
- **Волна 3:** `run/play.py`, `utils/__init__.py`, `run/__init__.py`

---

## Диаграмма зависимостей

```
Волна 1 (параллельно):
  state ──┐
  actions ─┼──► environment (Волна 2)
  rewards ─┘
  base_agent ──┬──► reinforce ──────┐
  policy ──────┼──► reinforce_baseline ──► trainer, run/* (Волна 3)
               └──► trpo ────────────┘
  config, utils, logger
```

---

## Чеклист перед каждой волной

**Волна 1:** формат state и action согласован (см. env/state.py, env/actions.py)

**Волна 2:** Person 1 сдал environment; Person 4 сдал base_agent и policy

**Волна 3:** env, agent, training готовы; можно запускать интеграционные тесты
