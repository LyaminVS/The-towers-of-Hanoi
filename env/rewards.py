"""
Расчёт наград и штрафов Tower of Hanoi.

Класс Reward позволяет легко менять настройки наград для экспериментов.
"""


class Reward:
    """
    Настраиваемая схема наград для среды Tower of Hanoi.
    
    Новая конфигурация по запросу пользователя:
        reward_step — штраф за каждый шаг (обычно -1)
        goal_penalty — штраф за состояние, где все диски на последнем стержне
        invalid_move — штраф за попытку положить больший диск на меньший
    """

    def __init__(
        self,
        reward_step: float = -1.0,
        goal_penalty: float = -10.0,
        invalid_move: float = -50.0,
        correct_placement: float = 10.0,
        use_correct_placement: bool = True,
    ):
        """
        Создать схему наград с заданными параметрами.
        
        Input:
            reward_step — штраф за каждый шаг (обычно отрицательный)
            goal_penalty — штраф, если все диски на последнем стержне
            invalid_move — штраф за недопустимый ход (больший на меньший)
            correct_placement — бонус за каждый диск на последнем стержне
            use_correct_placement — применяется ли бонус
        Output: —
        """
        self.reward_step = reward_step
        self.goal_penalty = goal_penalty
        self.invalid_move = invalid_move
        self.correct_placement = correct_placement
        self.use_correct_placement = use_correct_placement

    @classmethod
    def from_config(cls, config: dict) -> "Reward":
        """
        Создать Reward из словаря конфигурации (например, из config.settings).
        
        Input:
            config — dict с ключами: REWARD_STEP, REWARD_GOAL,
                    REWARD_INVALID_MOVE
        Output: экземпляр Reward
        """
        def _get(key: str, default):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        return cls(
            reward_step=_get("REWARD_STEP", -1.0),
            goal_penalty=_get("REWARD_GOAL", -10.0),
            invalid_move=_get("REWARD_INVALID_MOVE", -50.0),
            correct_placement=_get("REWARD_CORRECT_PLACEMENT", 10.0),
            use_correct_placement=_get("USE_CORRECT_PLACEMENT", True),
        )


    def compute(
        self,
        is_invalid: bool,
        all_correct: bool,
        correct_count: int = 0,
    ) -> tuple[float, bool]:
        """
        Вычислить суммарную награду за шаг по текущей схеме.
        
        Схема:
            total = reward_step
            если is_invalid: total += invalid_move
            иначе если all_correct: total += goal_penalty
        Возвращается кортеж (total_reward, done) — done всегда False
        """
        total = self.reward_step

        if is_invalid:
            total += self.invalid_move
            # invalid move does not end episode per new spec
            return total, False

        if self.use_correct_placement:
            total += self.correct_placement * correct_count

        if all_correct:
            total += self.goal_penalty

        return total, False
