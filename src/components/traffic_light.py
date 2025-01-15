# src/components/traffic_light.py

import pygame
import os


class TrafficLight:
    def __init__(self, x, y, state_durations, state):
        # Load traffic light images
        light_images_path = "assets/images/lights"
        self.images = {
            "red": pygame.image.load(os.path.join(light_images_path, "red.png")),
            "yellow": pygame.image.load(os.path.join(light_images_path, "yellow.png")),
            "green": pygame.image.load(os.path.join(light_images_path, "green.png")),
        }

        # Scale the images
        self.images = {key: pygame.transform.scale(img, (40, 80)) for key, img in self.images.items()}

        self.state = state
        self.image = self.images[self.state]
        self.rect = self.image.get_rect(topleft=(x, y))

        self.timer = 0  # Timer to switch states
        self.state_durations = state_durations

    def update(self, state_durations):
        self.state_durations = state_durations
        self.timer += 1
        if self.timer >= self.state_durations[self.state]:
            self.timer = 0
            self._switch_state()

    def _switch_state(self):
        if self.state == "red":
            self.state = "green"
        elif self.state == "green":
            self.state = "yellow"
        elif self.state == "yellow":
            self.state = "red"

        # Update the image
        self.image = self.images[self.state]

    def render(self, screen):
        screen.blit(self.image, self.rect)
