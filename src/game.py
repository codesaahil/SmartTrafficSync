import pygame
from utils.settings import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from systems.car_system import CarSystem
from systems.traffic_light_system import TrafficLightSystem

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Traffic Light Simulation")
        self.clock = pygame.time.Clock()
        self.running = True

        # Load the background image
        self.background = pygame.image.load("assets/images/intersection.jpeg")
        self.background = pygame.transform.scale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Initialize the traffic light system
        self.traffic_light_system = TrafficLightSystem(self.screen)

        # Initialize the car system, passing the traffic light system
        self.car_system = CarSystem(self.screen)

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
        self.traffic_light_system.update()
        self.car_system.update(self.traffic_light_system)

    def render(self):
        # Draw the background
        self.screen.blit(self.background, (0, 0))

        # Render the traffic lights
        self.traffic_light_system.render()

        # Render the cars
        self.car_system.render()

        pygame.display.flip()

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
