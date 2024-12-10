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

        # Initialize traffic lights
        for pos in self.positions:
            self.traffic_lights.append(TrafficLight(*pos))

    def update(self):
        for light in self.traffic_lights:
            light.update()

    def render(self):
        for light in self.traffic_lights:
            light.render(self.screen)
