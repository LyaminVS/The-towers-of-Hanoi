"""
Скрипт оценки обученного агента. Запуск: python run/evaluate.py [--args]
"""


def parse_args() -> object:
    """
    Парсинг аргументов командной строки.
    
    Input: —
    Output: объект с полями:
        num_disks — количество дисков
        agent_method — метод агента (должен совпадать с обученной моделью)
        load_model — путь к файлу модели
        render — визуализировать ли ход
        num_episodes — количество эпизодов для оценки
        save_results — путь для сохранения результатов (None — не сохранять)
        load_params — путь к JSON с параметрами (переопределяет остальные)
        save_params — путь для сохранения текущих параметров в JSON
    """
    ...


def evaluate(env, agent, num_episodes: int = 10, render: bool = False) -> dict:
    """
    Оценка обученного агента без exploration (greedy действия).
    
    Input:
        env — среда TowerOfHanoiEnv
        agent — агент с загруженной моделью (training=False для select_action)
        num_episodes — количество эпизодов для оценки
        render — если True, вызывать env.render() после каждого шага
    Output: dict со средними метриками:
        mean_reward — средняя награда за эпизод
        mean_steps — среднее количество шагов
        success_rate — доля успешных эпизодов (0..1)
    """
    ...


def main() -> None:
    """
    Запуск оценки: загрузить параметры (из --load_params или args), модель, вызвать evaluate().
    
    Порядок: parse_args → load_params (если --load_params) → create_env → create_agent →
            agent.load → evaluate → save_results (если --save_results)
    
    Input: —
    Output: None
    """
    ...


if __name__ == "__main__":
    main()
