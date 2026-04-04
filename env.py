class Observation:
    def __init__(self, email, step):
        self.goal = "Classify the email correctly"
        self.url = f"inbox://email/{step}"
        self.screenshot = None
        self.last_action_error = False

        self.metadata = {
            "browsergym_obs": {
                "extra_element_properties": {
                    "1": {"clickable": True},
                    "2": {"clickable": True},
                    "3": {"clickable": True},
                }
            }
        }

        # Email data
        self.email = {
            "subject": email["subject"],
            "body": email["body"],
            "sender": email["sender"],
            "timestamp": email["timestamp"]
        }

        # ✅ NEW: Priority score (based on keywords)
        urgent_keywords = ["urgent", "account", "payment", "alert"]
        score = 0

        for word in urgent_keywords:
            if word in email["subject"].lower():
                score += 1

        self.priority_score = min(score / 4, 1.0)


class Result:
    def __init__(self, observation, reward, done, info=None):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}


class EmailEnv:
    def __init__(self, difficulty="easy"):
        self.valid_actions = ["mark_spam", "mark_important", "mark_normal"]

        self.easy_emails = [
            {
                "subject": "URGENT: Account Suspended",
                "body": "Please verify your bank account immediately",
                "sender": "security@bank.com",
                "timestamp": "2026-04-01 09:00",
                "label": "important"
            },
            {
                "subject": "Win iPhone 15 Now!!!",
                "body": "Click here to claim your prize",
                "sender": "promo@spam.com",
                "timestamp": "2026-04-01 10:15",
                "label": "spam"
            },
            {
                "subject": "Team Meeting Tomorrow",
                "body": "Let’s meet at 10 AM",
                "sender": "manager@company.com",
                "timestamp": "2026-04-01 11:30",
                "label": "normal"
            }
        ]

        self.emails = self.easy_emails

        self.current_index = 0
        self.current_email = None
        self.done = False
        self.correct = 0

    def reset(self):
        self.current_index = 0
        self.done = False
        self.correct = 0
        self.current_email = self.emails[self.current_index]
        return Result(self.state(), 0, False)

    def parse_action(self, action_str):
        if "'1'" in action_str:
            return "mark_spam"
        elif "'2'" in action_str:
            return "mark_important"
        elif "'3'" in action_str:
            return "mark_normal"
        return "invalid"

    def step(self, action_str):
        if self.done:
            return Result(self.state(), 0, True)

        action = self.parse_action(action_str)
        correct_label = self.current_email["label"]

        # ✅ NEW: Confidence-based reward
        if action == f"mark_{correct_label}":
            confidence = 0.9
            reward = confidence
            self.correct += 1

        elif action == "invalid":
            confidence = 0.0
            reward = -1

        else:
            # partial correctness
            confidence = 0.3
            reward = -0.5 + confidence * 0.2

        # move forward
        self.current_index += 1

        if self.current_index >= len(self.emails):
            self.done = True
        else:
            self.current_email = self.emails[self.current_index]

        return Result(
            self.state(),
            reward,
            self.done,
            info={
                "confidence": confidence,
                "correct_label": correct_label
            }
        )

    def state(self):
        return Observation(self.current_email, self.current_index)

    def get_score(self):
        return self.correct / len(self.emails)
