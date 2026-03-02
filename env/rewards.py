"""
Расчёт наград и штрафов Tower of Hanoi.

Класс Reward позволяет легко менять настройки наград для экспериментов.
"""


class Reward:
    """
    Настраиваемая схема наград для среды Tower of Hanoi.
    
    Атрибуты (можно менять при создании или через конфиг):
        step — награда за каждый шаг (штраф за длину пути)
        invalid_move — штраф за недопустимый ход (диск на меньший)
        correct_placement — бонус за правильное размещение на 3-й палке
        death_penalty — большой штраф при нарушении (если use_death_penalty)
        use_correct_placement — выдавать ли бонус за correct_placement
        use_death_penalty — завершать эпизод и давать death_penalty при invalid
    """

    def __init__(
        self,
        step: float = -1.0,
        invalid_move: float = -10.0,
        correct_placement: float = 10.0,
        death_penalty: float = -100.0,
        use_correct_placement: bool = True,
        use_death_penalty: bool = False,
    ):
        """
        Создать схему наград с заданными параметрами.
        
        Input:
            step — награда за каждый шаг (отрицательная = штраф за длину пути)
            invalid_move — штраф за недопустимый ход (диск положен на меньший)
            correct_placement — бонус за размещение диска в правильной позиции
                               на третьей палке (по порядку снизу вверх)
            death_penalty — штраф при «смерти» (используется если use_death_penalty)
            use_correct_placement — если True, добавлять correct_placement при
                                   правильном размещении
            use_death_penalty — если True, при invalid ход даёт death_penalty
                               и эпизод завершается (done=True)
        Output: —
        """
        ...

    @classmethod
    def from_config(cls, config: dict) -> "Reward":
        """
        Создать Reward из словаря конфигурации (например, из config.settings).
        
        Input:
            config — dict с ключами: REWARD_STEP, REWARD_INVALID_MOVE,
                    REWARD_CORRECT_PLACEMENT, REWARD_DEATH,
                    USE_CORRECT_PLACEMENT, USE_DEATH_PENALTY
        Output: экземпляр Reward
        """
        ...

    def compute(self, is_invalid: bool, is_correct_placement: bool) -> tuple[float, bool]:
        """
        Вычислить суммарную награду за шаг.
        
        Формула: total = step + (death_penalty если is_invalid и use_death_penalty,
                иначе invalid_move если is_invalid) + (correct_placement если
                is_correct_placement и use_correct_placement)
        
        Input:
            is_invalid — был ли ход недопустимым (диск на меньший)
            is_correct_placement — было ли правильное размещение на 3-й палке
        Output: (total_reward, done)
            total_reward — суммарная награда за шаг
            done — True если use_death_penalty и is_invalid (эпизод завершён)
        """
        ...
