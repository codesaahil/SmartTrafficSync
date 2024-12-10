import pygame

class RenderSystem:
    def __init__(self, screen):
        self.screen = screen

    def render(self, obj):
        pygame.draw.rect(self.screen, obj.color, obj.rect)
