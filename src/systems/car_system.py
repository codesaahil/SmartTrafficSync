import random
from components.car import Car

class CarSystem:
    def __init__(self, screen):
        self.screen = screen
        self.cars = []  # List to hold active car instances
        self.car_count = {"up": 0, "down": 0, "left": 0, "right": 0}  # Global count of cars per direction
        self.spawn_timer = 0  # Timer to control spawning
        self.spawn_interval = random.randint(180, 360)  # Random interval in frames (0.5-1.5 seconds at 60 FPS)

        # Define stop positions for each direction
        self.stop_positions = {
            "right": 380,
            "left": 620,
            "up": 620,
            "down": 380,
        }

    def spawn_car(self):
        directions = ["up", "down", "left", "right"]
        max_attempts = 10  # To avoid infinite loops when placing cars
        for _ in range(max_attempts):
            direction = random.choice(directions)

            # Set the initial position and movement direction based on spawn direction
            if direction == "up":
                x, y = self.screen.get_width() // 2 - 50, self.screen.get_height()
                dx, dy = 0, -1
            elif direction == "down":
                x, y = self.screen.get_width() // 2 + 50, -80
                dx, dy = 0, 1
            elif direction == "left":
                x, y = self.screen.get_width(), self.screen.get_height() // 2 - 50
                dx, dy = -1, 0
            elif direction == "right":
                x, y = -80, self.screen.get_height() // 2 + 50
                dx, dy = 1, 0

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

    def update(self, traffic_light_system):
        # Increment the spawn timer
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_car()
            self.spawn_timer = 0
            self.spawn_interval = random.randint(30, 90)  # Reset with a new random interval

        for car in self.cars[:]:
            car.is_stopped = False
            stop_position = self.stop_positions[car.direction]
            relevant_lights = self._get_relevant_lights(car.direction)
            
            # Check traffic light state
            should_stop_for_light = any(
                traffic_light_system.traffic_lights[light_index].state in ["red", "yellow"]
                for light_index in relevant_lights
            )

            # Check for cars ahead
            should_stop_for_car = self._should_stop_for_car_ahead(car)

            # Combine stopping conditions
            if (should_stop_for_light and self._is_at_stop_position(car, stop_position)) or should_stop_for_car:
                car.is_stopped = True

            if car.is_stopped:
                continue

            # Move the car
            car.rect.x += car.dx
            car.rect.y += car.dy

            # Remove the car if it goes off-screen
            if (
                car.rect.right < 0
                or car.rect.left > self.screen.get_width()
                or car.rect.bottom < 0
                or car.rect.top > self.screen.get_height()
            ):
                self.cars.remove(car)
                self.car_count[car.direction] -= 1

    def _get_relevant_lights(self, direction):
        if direction == "right":
            return [0, 4]
        if direction == "left":
            return [3, 7]
        if direction == "up":
            return [5, 6]
        if direction == "down":
            return [1, 2]
        return []

    def _is_at_stop_position(self, car, stop_position):
        if car.direction == "right":
            return car.rect.x + car.rect.width == stop_position
        if car.direction == "left":
            return car.rect.x == stop_position
        if car.direction == "up":
            return car.rect.y == stop_position
        if car.direction == "down":
            return car.rect.y + car.rect.height == stop_position
        return False
    
    def _should_stop_for_car_ahead(self, car):
        cars_in_direction = [
            other_car for other_car in self.cars 
            if other_car.direction == car.direction and other_car != car
        ]
        
        # Sort cars in the direction of movement
        if car.direction in ["right", "down"]:
            cars_in_direction.sort(key=lambda x: (x.rect.x, x.rect.y))
        else:  # left, up
            cars_in_direction.sort(key=lambda x: (-x.rect.x, -x.rect.y), reverse=True)

        # Check if there's a car ahead that requires stopping
        for other_car in cars_in_direction:
            # Calculate the distance between cars based on their direction
            if car.direction == "right":
                distance = other_car.rect.x - (car.rect.x + car.rect.width)
            elif car.direction == "left":
                distance = car.rect.x - (other_car.rect.x + other_car.rect.width)
            elif car.direction == "down":
                distance = other_car.rect.y - (car.rect.y + car.rect.height)
            elif car.direction == "up":
                distance = car.rect.y - (other_car.rect.y + other_car.rect.height)
            
            # Stop if another car is too close (less than 20 pixels)
            if 0 <= distance < 20:
                return True
        
        return False

    def render(self):
        for car in self.cars:
            car.render(self.screen)
