import random

class PPO:
    @staticmethod
    def predict(processed):
        action = random.uniform(-1, 1)  # e.g., buy/sell signal
        return {"action": action, "reason": processed['rationale']}
