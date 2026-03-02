"""
Визуализация среды Tower of Hanoi с использованием Pygame.
"""

import os
import pygame
import colorsys

# Скрываем приветственное сообщение Pygame в консоли
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class PygameRenderer:
    """
    Класс для отрисовки Ханойской башни в отдельном окне.
    """

    # Цветовая палитра (современная темная тема)
    BG_COLOR = (30, 30, 36)
    BASE_COLOR = (80, 80, 90)
    STICK_COLOR = (120, 120, 130)
    TEXT_COLOR = (220, 220, 220)
    HIGHLIGHT_COLOR = (255, 215, 0)  # Золотой для выделения
    ERROR_COLOR = (255, 100, 100)
    SUCCESS_COLOR = (100, 255, 100)

    def __init__(self, num_disks: int, num_sticks: int = 3):
        self.num_disks = num_disks
        self.num_sticks = num_sticks

        # Настройки размеров
        self.stick_spacing = 250
        self.base_height = 40
        self.stick_width = 16
        self.disk_height = 30
        self.max_disk_width = 200
        self.min_disk_width = 60

        # Размер окна
        self.width = self.stick_spacing * self.num_sticks
        self.height = self.num_disks * self.disk_height + 250

        # Инициализация Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Tower of Hanoi - {self.num_disks} Disks")
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 18)

        self.clock = pygame.time.Clock()
        self.disk_colors = self._generate_colors(self.num_disks)

    def _generate_colors(self, num_colors: int) -> list:
        """Генерирует красивый градиент цветов для дисков (от красного к фиолетовому)."""
        colors = []
        for i in range(num_colors):
            hue = i / max(1, (num_colors - 1)) * 0.8
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def render(self, state: tuple, step_count: int, total_reward: float, selected_stick: int = None, message: str = ""):
        """
        Отрисовать текущее состояние.
        """
        self.screen.fill(self.BG_COLOR)

        # Рисуем нижнюю базу
        base_rect = pygame.Rect(
            50, self.height - self.base_height - 50,
            self.width - 100, self.base_height
        )
        pygame.draw.rect(self.screen, self.BASE_COLOR, base_rect, border_radius=10)

        # Рисуем палки
        stick_y_start = self.height - self.base_height - 50 - (self.num_disks + 2) * self.disk_height
        stick_height_total = (self.num_disks + 2) * self.disk_height

        for i in range(self.num_sticks):
            stick_x = self._get_stick_center_x(i)
            stick_rect = pygame.Rect(
                stick_x - self.stick_width // 2, stick_y_start,
                self.stick_width, stick_height_total
            )
            
            # Подсветка выбранной палки
            color = self.HIGHLIGHT_COLOR if i == selected_stick else self.STICK_COLOR
            pygame.draw.rect(self.screen, color, stick_rect, border_radius=8)

            # Номера палок под базой
            text = self.font.render(str(i + 1), True, self.TEXT_COLOR)
            self.screen.blit(text, (stick_x - text.get_width() // 2, self.height - 40))

        # Рисуем диски (0 — самый большой)
        for disk_idx, (stick, height) in enumerate(state):
            width_step = (self.max_disk_width - self.min_disk_width) / max(1, self.num_disks - 1)
            disk_width = self.max_disk_width - (disk_idx * width_step)
            
            center_x = self._get_stick_center_x(stick)
            y = self.height - self.base_height - 50 - (height + 1) * self.disk_height

            rect = pygame.Rect(0, 0, disk_width, self.disk_height - 2)
            rect.center = (center_x, y + self.disk_height // 2)

            pygame.draw.rect(self.screen, self.disk_colors[disk_idx], rect, border_radius=12)

        # Текстовая информация (Шаги и Награда)
        info_text = self.font.render(f"Steps: {step_count}   |   Score: {total_reward:.1f}", True, self.TEXT_COLOR)
        self.screen.blit(info_text, (20, 20))

        # Сообщения (Ошибки, Подсказки)
        if message:
            if "Invalid" in message or "Penalty" in message:
                msg_color = self.ERROR_COLOR
            elif "Reward" in message or "Good" in message or "Perfect" in message:
                msg_color = self.SUCCESS_COLOR
            else:
                msg_color = self.HIGHLIGHT_COLOR
            
            msg_text = self.font.render(message, True, msg_color)
            self.screen.blit(msg_text, (self.width // 2 - msg_text.get_width() // 2, 20))

        # Подсказка по управлению
        help_text = self.small_font.render("Click sticks to move, or use keys 1, 2, 3. ESC to quit.", True, self.BASE_COLOR)
        self.screen.blit(help_text, (self.width // 2 - help_text.get_width() // 2, self.height - 25))

        pygame.display.flip()

    def _get_stick_center_x(self, stick_idx: int) -> int:
        return int((stick_idx + 0.5) * self.stick_spacing)

    def get_human_action(self, state: tuple, step_count: int, total_reward: float, valid_actions: list, initial_message: str = "") -> tuple:
        """
        Игровой цикл ожидания хода от человека.
        """
        selected_stick = None
        message = initial_message

        while True:
            self.render(state, step_count, total_reward, selected_stick, message)
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    
                    stick_clicked = None
                    if event.key in (pygame.K_1, pygame.K_KP1): stick_clicked = 0
                    if event.key in (pygame.K_2, pygame.K_KP2): stick_clicked = 1
                    if event.key in (pygame.K_3, pygame.K_KP3): stick_clicked = 2

                    if stick_clicked is not None:
                        res, selected_stick, message = self._process_selection(stick_clicked, selected_stick, valid_actions)
                        if res is not None:
                            return res

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_x, _ = pygame.mouse.get_pos()
                    stick_clicked = mouse_x // self.stick_spacing
                    
                    if 0 <= stick_clicked < self.num_sticks:
                        res, selected_stick, message = self._process_selection(stick_clicked, selected_stick, valid_actions)
                        if res is not None:
                            return res

    def _process_selection(self, stick_clicked: int, selected_stick: int, valid_actions: list):
        if selected_stick is None:
            # Проверяем, есть ли на палке диски (можно ли сделать ход с неё)
            if any(action[0] == stick_clicked for action in valid_actions):
                return None, stick_clicked, "Select target stick..."
            else:
                return None, None, "Invalid! Stick is empty."
        else:
            if selected_stick == stick_clicked:
                return None, None, "Selection canceled."
            
            action = (selected_stick, stick_clicked)
            # Если ход в списке доступных (в среде он доступен, даже если диск больше — это даст штраф)
            if action in valid_actions:
                return action, None, ""
            else:
                return None, None, "Invalid target!"

    def show_end_screen(self, steps: int, total_reward: float, is_victory: bool):
        """Экран конца игры (Победа или Смерть)."""
        self.screen.fill(self.BG_COLOR)
        
        main_text = f"VICTORY in {steps} steps!" if is_victory else "GAME OVER!"
        color = self.SUCCESS_COLOR if is_victory else self.ERROR_COLOR
        
        vic_text = self.font.render(main_text, True, color)
        score_text = self.font.render(f"Final Score: {total_reward:.1f}", True, self.HIGHLIGHT_COLOR)
        close_text = self.small_font.render("Press any key or click to exit...", True, self.TEXT_COLOR)
        
        self.screen.blit(vic_text, (self.width // 2 - vic_text.get_width() // 2, self.height // 2 - 40))
        self.screen.blit(score_text, (self.width // 2 - score_text.get_width() // 2, self.height // 2))
        self.screen.blit(close_text, (self.width // 2 - close_text.get_width() // 2, self.height // 2 + 50))
        pygame.display.flip()

        waiting = True
        while waiting:
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    waiting = False

    def close(self):
        pygame.quit()