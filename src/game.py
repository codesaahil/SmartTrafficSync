# src/game.py
import pygame
from utils.settings import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from systems.car_system import CarSystem

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Car Game")
        self.clock = pygame.time.Clock()
        self.running = True

        # Load the background image
        self.background = pygame.image.load("assets/images/intersection.jpeg")
        self.background = pygame.transform.scale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Initialize the car system
        self.car_system = CarSystem(self.screen)
        self.spawn_timer = 0  # Timer for spawning cars

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        # Spawn a car every 2 seconds
        self.spawn_timer += 1
        if self.spawn_timer >= FPS * 2:
            self.car_system.spawn_car()
            self.spawn_timer = 0

        # Update the car system
        self.car_system.update()

    def render(self):
        # Draw the background
        self.screen.blit(self.background, (0, 0))

        # Render the cars
        self.car_system.render()

        # Show car counts (optional for debugging)
        font = pygame.font.Font(None, 36)
        count_text = f"Up: {self.car_system.car_count['up']} Down: {self.car_system.car_count['down']} Left: {self.car_system.car_count['left']} Right: {self.car_system.car_count['right']}"
        text_surface = font.render(count_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
