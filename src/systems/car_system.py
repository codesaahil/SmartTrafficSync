import random
from components.car import Car

class CarSystem:
    def __init__(self, screen):
        self.screen = screen
        self.cars = []  # List to hold active car instances
        self.car_count = {"up": 0, "down": 0, "left": 0, "right": 0}  # Global count of cars per direction
        self.spawn_timer = 0  # Timer to control spawning
        self.spawn_interval = random.randint(30, 90)  # Random interval in frames (0.5-1.5 seconds at 60 FPS)

    def spawn_car(self):
        directions = ["up", "down", "left", "right"]
        max_attempts = 10  # To avoid infinite loops when placing cars
        for _ in range(max_attempts):
            direction = random.choice(directions)

            # Set the initial position and movement direction based on spawn direction
            if direction == "up":
                x, y = self.screen.get_width() // 2 - 50, self.screen.get_height()
                dx, dy = 0, -5
            elif direction == "down":
                x, y = self.screen.get_width() // 2 + 50, -80
                dx, dy = 0, 5
            elif direction == "left":
                x, y = self.screen.get_width(), self.screen.get_height() // 2 - 50
                dx, dy = -5, 0
            elif direction == "right":
                x, y = -80, self.screen.get_height() // 2 + 50
                dx, dy = 5, 0

            # Create a new car object
            new_car = Car(x, y, direction)
            new_car.direction = direction
            new_car.dx = dx
            new_car.dy = dy

            # Ensure the new car does not overlap with existing cars
            if not any(new_car.rect.colliderect(existing_car.rect) for existing_car in self.cars):
                self.cars.append(new_car)
                self.car_count[direction] += 1
                break  # Exit after successful spawn

    def update(self):
        # Increment the spawn timer
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_car()
            self.spawn_timer = 0
            self.spawn_interval = random.randint(30, 90)  # Reset with a new random interval

        # Update car positions and remove off-screen cars
        for car in self.cars[:]:
            car.rect.x += car.dx
            car.rect.y += car.dy

            if (
                car.rect.right < 0
                or car.rect.left > self.screen.get_width()
                or car.rect.bottom < 0
                or car.rect.top > self.screen.get_height()
            ):
                self.cars.remove(car)
                self.car_count[car.direction] -= 1

    def render(self):
        for car in self.cars:
            car.render(self.screen)
