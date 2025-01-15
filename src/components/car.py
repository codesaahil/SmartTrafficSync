# src/components/car.py

import os
import random
import pygame

class Car:
    def __init__(self, x, y, direction):
        # Load a random car image from assets/images/cars
        car_images_path = "assets/images/cars"
        car_images = [os.path.join(car_images_path, img) for img in os.listdir(car_images_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

        if not car_images:
            raise FileNotFoundError("No car images found in assets/images/cars")

        # Randomly select and load an image
        self.image = pygame.image.load(random.choice(car_images))
        self.image = pygame.transform.scale(self.image, (80, 40))  # Scale the image to a standard size

        # Rotate the image based on the direction
        if direction == "up":
            self.image = pygame.transform.rotate(self.image, 90)
        elif direction == "down":
            self.image = pygame.transform.rotate(self.image, -90)
        elif direction == "left":
            self.image = pygame.transform.rotate(self.image, 180)
        # "right" doesn't need rotation

        self.is_stopped = False

        # Store the position and rectangle
        self.rect = self.image.get_rect(topleft=(x, y))

    def render(self, screen):
        screen.blit(self.image, self.rect)
