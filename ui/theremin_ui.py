"""
ui/theremin_ui.py — Theremin mode spatial visualization.

Shows hand position, pitch/volume indicators, and frequency display.

TODO (Collaborator B): Implement theremin visualization.
"""

import pygame


class ThereminUI:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        pygame.font.init()
        self.font_medium = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI", 28, bold=True)

    def draw(self, surface, theremin_data: dict, pinch_pos=None):
        """
        Draw theremode UI overlay.

        Args:
            surface: Pygame surface to draw on.
            theremin_data: {pitch, volume, engaged, filter_cutoff}
            pinch_pos: (x, y) of pinch point in pixel coords, or None.
        """
        freq = theremin_data.get("pitch", 0)
        vol = theremin_data.get("volume", 0)
        engaged = theremin_data.get("engaged", False)
        cutoff = theremin_data.get("filter_cutoff", 1.0)

        # --- Background panel ---
        panel = pygame.Surface((200, 100), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 120))
        surface.blit(panel, (self.width - 220, 20))

        # --- Frequency display ---
        freq_color = (100, 255, 100) if engaged else (150, 150, 150)
        freq_lbl = self.font_large.render(f"{freq:.1f} Hz", True, freq_color)
        surface.blit(freq_lbl, (self.width - 210, 25))

        # --- Volume bar ---
        vol_text = self.font_medium.render(f"Vol: {vol:.0%}", True, (255, 255, 255))
        surface.blit(vol_text, (self.width - 210, 60))
        bar_x, bar_y, bar_w, bar_h = self.width - 210, 80, 180, 12
        pygame.draw.rect(surface, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        fill_w = int(bar_w * vol)
        if fill_w > 0:
            pygame.draw.rect(surface, (100, 200, 255), (bar_x, bar_y, fill_w, bar_h), border_radius=3)

        # --- Filter cutoff bar ---
        cutoff_text = self.font_medium.render(f"Bright: {cutoff:.0%}", True, (255, 255, 255))
        surface.blit(cutoff_text, (self.width - 210, 95))
        fill_w = int(bar_w * cutoff)
        if fill_w > 0:
            pygame.draw.rect(surface, (255, 180, 50), (bar_x, bar_y + 16, fill_w, bar_h), border_radius=3)

        # --- Pinch indicator ---
        if pinch_pos:
            px, py = pinch_pos
            color = (0, 255, 100) if engaged else (255, 80, 80)
            pygame.draw.circle(surface, color, (px, py), 15, 3)
            status = "SOUND ON" if engaged else "SOUND OFF"
            status_lbl = self.font_medium.render(status, True, color)
            surface.blit(status_lbl, (px - status_lbl.get_width() // 2, py + 20))

        # --- Engagement ring around cursor ---
        if engaged and pinch_pos:
            px, py = pinch_pos
            t = pygame.time.get_ticks() / 1000.0
            radius = 20 + int(5 * abs((t % 1.0) - 0.5))
            pygame.draw.circle(surface, (0, 255, 150, 100), (px, py), radius, 2)
