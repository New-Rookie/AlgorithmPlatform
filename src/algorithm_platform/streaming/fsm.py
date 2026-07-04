class SOPFSM:
    def __init__(self, steps, threshold=0.7):
        self.steps = steps
        self.threshold = threshold
        self.index = 0
        self.history = []

    def current_step(self):
        return self.steps[self.index] if self.index < len(self.steps) else None

    def update(self, action, confidence, timestamp):
        step = self.current_step()
        if step is None:
            return "FINISHED"

        expected_actions = step.get("actions", [])

        if confidence < self.threshold:
            return "UNCERTAIN"

        if action in expected_actions:
            self.history.append((timestamp, action, step["id"]))
            self.index += 1
            return "OK"

        # possible skip detection
        return "DEVIATION"
