"""
Расчёт наград и штрафов Tower of Hanoi.

Класс Reward позволяет легко менять настройки наград для экспериментов.
"""


class Reward:
    """
    Настраиваемая схема наград для среды Tower of Hanoi.
    
    Атрибуты (можно менять при создании или через конфиг):
        reward_step — награда за каждый шаг (штраф за длину пути)
        invalid_move — штраф за недопустимый ход (диск на меньший)
        correct_placement — бонус за правильное размещение на 3-й палке
        death_penalty — большой штраф при нарушении (если use_death_penalty)
        use_correct_placement — выдавать ли бонус за correct_placement
        use_death_penalty — завершать эпизод и давать death_penalty при invalid
    """

    def __init__(
        self,
        reward_step: float = -1.0,
        invalid_move: float = -10.0,
        correct_placement: float = 10.0,
        death_penalty: float = -100.0,
        use_correct_placement: bool = True,
        use_death_penalty: bool = False,
    ):
        """
        Создать схему наград с заданными параметрами.
        
        Input:
            reward_step — награда за каждый шаг (отрицательная = штраф за длину пути)
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
        self.reward_step = reward_step
        self.invalid_move = invalid_move
        self.correct_placement = correct_placement
        self.death_penalty = death_penalty
        self.use_correct_placement = use_correct_placement
        self.use_death_penalty = use_death_penalty

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
        def _get(key: str, default):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        return cls(
            reward_step=_get("REWARD_STEP", -1.0),
            invalid_move=_get("REWARD_INVALID_MOVE", -10.0),
            correct_placement=_get("REWARD_CORRECT_PLACEMENT", 10.0),
            death_penalty=_get("REWARD_DEATH", -100.0),
            use_correct_placement=_get("USE_CORRECT_PLACEMENT", True),
            use_death_penalty=_get("USE_DEATH_PENALTY", False),
        )

    def compute(
        self,
        is_invalid: bool,
        is_correct_placement: list[bool],
        max_steps: int | None = None,
        step_number: int | None = None,
    ) -> tuple[float, bool]:
        """
        Вычислить суммарную награду за шаг.
        
        Формула: total = reward_step + (death_penalty + (max_steps - step_number) если
                is_invalid и use_death_penalty — штраф за несхоженные шаги,
                иначе invalid_move если is_invalid + correct_placement * count
                для каждого True в is_correct_placement
        
        Input:
            is_invalid — был ли ход недопустимым (диск на меньший)
            is_correct_placement — list[bool]: для каждого диска — в правильной
                                  ли позиции на 3-й палке (индекс = номер диска)
            max_steps — макс. шагов в эпизоде (нужно при use_death_penalty)
            step_number — номер текущего шага (1-based, после хода)
        Output: (total_reward, done)
            total_reward — суммарная награда за шаг
            done — True если use_death_penalty и is_invalid (эпизод завершён)
        """
        total = self.reward_step

        if is_invalid:
            if self.use_death_penalty:
                extra = 0
                if max_steps is not None and step_number is not None:
                    # Штраф за несхоженные шаги: (max_steps - step_number) * reward_step
                    extra = (max_steps - step_number) * self.reward_step
                total += self.death_penalty + extra
                return total, True
            total += self.invalid_move

        total += self.correct_placement * sum(is_correct_placement)

        # При победе: начислить награду за все «виртуальные» оставшиеся шаги (как будто остаёмся в победе)
        if (all(is_correct_placement) and max_steps is not None and step_number is not None
                and step_number < max_steps):
            total += (max_steps - step_number) * (
                sum(is_correct_placement) * self.correct_placement - self.reward_step
            )

        return total, False
