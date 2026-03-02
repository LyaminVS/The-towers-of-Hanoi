"""
Среда Tower of Hanoi (Gym-like интерфейс).

State space: [палка каждого диска, высота каждого диска]
Action space: переместить один из верхних дисков на другую палку
"""


class TowerOfHanoiEnv:
    """
    Среда «Ханойская башня» для RL.
    
    Атрибуты:
        num_disks: int — количество дисков
        num_sticks: int — количество палок (всегда 3)
        max_steps: int — лимит шагов на эпизод (для truncated)
        reward: Reward — схема наград (настраиваемая)
    """

    def __init__(
        self,
        num_disks: int = 3,
        num_sticks: int = 3,
        max_steps: int = 100,
        reward: "Reward | None" = None,
    ):
        """
        Инициализация среды.
        
        Input:
            num_disks — количество дисков (по умолчанию 3)
            num_sticks — количество палок (обычно 3)
            max_steps — максимальное число шагов за эпизод; при превышении
                       step() вернёт truncated=True
            reward — экземпляр Reward; если None, создаётся с дефолтными значениями
        Output: —
        """
        ...

    def reset(self) -> tuple:
        """
        Сброс среды в начальное состояние (все диски на первой палке).
        
        Input: —
        Output: (observation, info)
            observation — начальное наблюдение в формате для агента
            info — dict: state (внутр. состояние), step_count (0), и др.
        """
        ...

    def step(self, action: tuple) -> tuple:
        """
        Выполнить действие и перейти в новое состояние.
        
        Input:
            action — (from_stick, to_stick), индексы палок 0..2
        Output: (observation, reward, terminated, truncated, info)
            observation — новое наблюдение после хода
            reward — награда (по схеме Reward: step, invalid, correct, death)
            terminated — True если цель достигнута (все на 3-й палке) или
                        смерть (use_death_penalty и invalid)
            truncated — True если step_count >= max_steps
            info — dict: is_invalid, is_correct_placement, step_count, state, ...
        """
        ...

    def render(self, mode: str = "human") -> str | None:
        """
        Визуализация текущего состояния (ASCII-арт).
        
        Input:
            mode — "human": вывести в stdout и вернуть None;
                   "ansi": вернуть строку с визуализацией
        Output: None при mode="human", str при mode="ansi"
        """
        ...

    def get_valid_actions(self) -> list:
        """
        Получить допустимые действия в текущем состоянии.
        
        Input: —
        Output: список [(from_stick, to_stick), ...] — только валидные ходы
        """
        ...

    def get_action_space(self) -> list:
        """
        Получить полное пространство действий (все пары палок).
        
        Input: —
        Output: список всех действий [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
        """
        ...


def create_env(
    num_disks: int = 3,
    num_sticks: int = 3,
    max_steps: int = 100,
    reward: "Reward | None" = None,
) -> TowerOfHanoiEnv:
    """
    Фабрика: создать экземпляр среды.
    
    Input:
        num_disks, num_sticks, max_steps — параметры среды
        reward — опционально, экземпляр Reward
    Output: экземпляр TowerOfHanoiEnv
    """
    ...
