"""
Визуализация среды Tower of Hanoi с использованием Pygame.
"""

import os
import pygame
import colorsys

# Скрываем приветственное сообщение Pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

class PygameRenderer:
    # Цветовая палитра
    BG_COLOR = (25, 25, 30)
    BASE_COLOR = (70, 70, 80)
    STICK_COLOR = (100, 100, 110)
    TEXT_COLOR = (240, 240, 240)
    HIGHLIGHT_COLOR = (255, 215, 0)
    ERROR_COLOR = (255, 80, 80)
    SUCCESS_COLOR = (80, 255, 150)

    def __init__(self, num_disks: int, num_sticks: int = 3):
        self.num_disks = num_disks
        self.num_sticks = num_sticks

        # --- УВЕЛИЧЕННЫЕ РАЗМЕРЫ ДЛЯ БОЛЬШОГО ОКНА ---
        self.stick_spacing = 350  # Было 250
        self.base_height = 50
        self.stick_width = 20
        self.disk_height = 35     # Было 30
        self.max_disk_width = 280 # Было 200
        self.min_disk_width = 80
        
        # Динамический размер окна
        self.width = self.stick_spacing * self.num_sticks
        self.height = max(600, self.num_disks * self.disk_height + 400)

        pygame.init()
        # Попытка сделать окно более четким на HighDPI мониторах
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Tower of Hanoi - {self.num_disks} Disks")
        
        # Шрифты
        self.font_main = pygame.font.SysFont("Verdana", 28, bold=True)
        self.font_msg = pygame.font.SysFont("Verdana", 22, italic=True)
        self.font_small = pygame.font.SysFont("Verdana", 16)

        self.clock = pygame.time.Clock()
        self.disk_colors = self._generate_colors(self.num_disks)

    def _generate_colors(self, n: int) -> list:
        return [tuple(int(c*255) for c in colorsys.hsv_to_rgb(i/max(1, n-1)*0.7, 0.8, 0.9)) for i in range(n)]

    def render(self, state: tuple, step_count: int, total_reward: float, selected_stick: int = None, message: str = ""):
        self.screen.fill(self.BG_COLOR)

        # 1. ОТРИСОВКА ТЕКСТА (Разделяем по высоте, чтобы не перекрывались)
        # Верхняя строка: Шаги и Счет
        score_surface = self.font_main.render(f"STEPS: {step_count}   SCORE: {total_reward:.1f}", True, self.TEXT_COLOR)
        self.screen.blit(score_surface, (30, 30))

        # Вторая строка: Сообщения (опускаем ниже на y=80)
        if message:
            color = self.HIGHLIGHT_COLOR
            if "Invalid" in message or "Penalty" in message: color = self.ERROR_COLOR
            if "Good" in message or "Perfect" in message: color = self.SUCCESS_COLOR
            
            msg_surface = self.font_msg.render(message, True, color)
            # Центрируем сообщение
            msg_rect = msg_surface.get_rect(center=(self.width // 2, 90))
            self.screen.blit(msg_surface, msg_rect)

        # 2. БАЗА И ПАЛКИ
        base_y = self.height - 100
        pygame.draw.rect(self.screen, self.BASE_COLOR, (50, base_y, self.width - 100, self.base_height), border_radius=15)

        stick_h = (self.num_disks + 3) * self.disk_height
        for i in range(self.num_sticks):
            x = int((i + 0.5) * self.stick_spacing)
            color = self.HIGHLIGHT_COLOR if i == selected_stick else self.STICK_COLOR
            pygame.draw.rect(self.screen, color, (x - self.stick_width//2, base_y - stick_h, self.stick_width, stick_h), border_radius=10)
            
            # Номера палок
            num_surf = self.font_main.render(str(i+1), True, self.BASE_COLOR)
            self.screen.blit(num_surf, (x - num_surf.get_width()//2, base_y + 60))

        # 3. ДИСКИ
        for disk_idx, (stick, height) in enumerate(state):
            w_step = (self.max_disk_width - self.min_disk_width) / max(1, self.num_disks - 1)
            w = self.max_disk_width - (disk_idx * w_step)
            x = int((stick + 0.5) * self.stick_spacing)
            y = base_y - (height + 1) * self.disk_height
            
            rect = pygame.Rect(0, 0, w, self.disk_height - 4)
            rect.center = (x, y + self.disk_height // 2)
            pygame.draw.rect(self.screen, self.disk_colors[disk_idx], rect, border_radius=10)

        # Подсказка снизу
        help_surf = self.font_small.render("Controls: Mouse clicks or Keys [1], [2], [3] | ESC to Quit", True, self.STICK_COLOR)
        self.screen.blit(help_surf, (self.width // 2 - help_surf.get_width() // 2, self.height - 30))

        pygame.display.flip()

    def get_surface_bytes(self) -> bytes:
        """Вернуть текущий кадр как RGB bytes для сохранения в GIF/PNG."""
        return pygame.image.tostring(self.screen, "RGB", False)

    def get_human_action(self, state, steps, reward, valid_actions, initial_msg):
        selected = None
        message = initial_msg
        while True:
            self.render(state, steps, reward, selected, message)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: return None
                    s_key = None
                    if event.key in (pygame.K_1, pygame.K_KP1): s_key = 0
                    if event.key in (pygame.K_2, pygame.K_KP2): s_key = 1
                    if event.key in (pygame.K_3, pygame.K_KP3): s_key = 2
                    if s_key is not None:
                        res, selected, message = self._process_selection(s_key, selected, valid_actions)
                        if res: return res
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    s_mouse = pygame.mouse.get_pos()[0] // self.stick_spacing
                    if 0 <= s_mouse < self.num_sticks:
                        res, selected, message = self._process_selection(s_mouse, selected, valid_actions)
                        if res: return res
            self.clock.tick(60)

    def _process_selection(self, clicked, selected, valid):
        if selected is None:
            if any(a[0] == clicked for a in valid): return None, clicked, "Where to move?"
            return None, None, "That stick is empty!"
        if selected == clicked: return None, None, "Canceled."
        action = (selected, clicked)
        if action in valid: return action, None, ""
        return None, None, "Invalid target stick!"

    def show_end_screen(self, steps, reward, win):
        self.screen.fill(self.BG_COLOR)
        txt = "VICTORY!" if win else "GAME OVER"
        clr = self.SUCCESS_COLOR if win else self.ERROR_COLOR
        surf = self.font_main.render(f"{txt} Steps: {steps} Score: {reward:.1f}", True, clr)
        self.screen.blit(surf, (self.width//2 - surf.get_width()//2, self.height//2))
        pygame.display.flip()
        pygame.time.wait(2000)

    def close(self):
        pygame.quit()