import random
from components.traffic_light import TrafficLight


class TrafficLightSystem:
    def __init__(self, screen):
        self.screen = screen
        self.traffic_lights = []

        # Positions for 8 traffic lights (2 per corner)
        self.positions = [
            (240, 280), (320, 180),  # Top-left corner
            (640, 180), (720, 280),  # Top-right corner
            (240, 640), (320, 740),  # Bottom-left corner
            (640, 740), (720, 640),  # Bottom-right corner
        ]

        self.state_durations = [{color: random.randint(1, 100) for color in ["red", "yellow", "green"]} for _ in range(8)]

        # Initialize traffic lights
        for (index, pos) in enumerate(self.positions):
            state = "red" if index in [0,3,4,7] else "green"
            self.traffic_lights.append(TrafficLight(*pos, self.state_durations[index], state))

    def update(self, state_durations):
        self.state_durations = state_durations
        for (index, light) in enumerate(self.traffic_lights):
            light.update(self.state_durations[index])

    def render(self):
        for light in self.traffic_lights:
            light.render(self.screen)
