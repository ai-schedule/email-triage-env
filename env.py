class Observation:
    def __init__(self, email, step, stage, history):
        self.goal = "Classify and take action on the email intelligently"
        self.url = f"inbox://email/{step}"
        self.screenshot = None
        self.last_action_error = False

        if stage == "classify":
            elements = {"1": {"clickable": True}, "2": {"clickable": True}, "3": {"clickable": True}}
        else:
            elements = {"4": {"clickable": True}, "5": {"clickable": True}, "6": {"clickable": True}}

        self.metadata = {"browsergym_obs": {"extra_element_properties": elements}}

        self.email = email
        self.stage = stage
        self.history = history[-3:]

        self.available_actions = list(elements.keys())

        urgent_keywords = ["urgent", "account", "payment", "alert"]
        score = sum(word in email["subject"].lower() for word in urgent_keywords)
        self.priority_score = min(score / 4, 1.0)


class Result:
    def __init__(self, observation, reward, done, info=None):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}


class EmailEnv:
    def __init__(self):
        self.valid_class_actions = ["mark_spam", "mark_important", "mark_normal"]
        self.valid_decision_actions = ["archive", "reply", "ignore"]

        self.emails = [
            {"subject": "URGENT: Account Suspended", "body": "Verify immediately", "sender": "bank@secure.com", "timestamp": "2026-04-01", "label": "important"},
            {"subject": "Win iPhone Now!!!", "body": "Click to claim", "sender": "spam@promo.com", "timestamp": "2026-04-01", "label": "spam"},
            {"subject": "Team Meeting", "body": "Tomorrow 10AM", "sender": "manager@company.com", "timestamp": "2026-04-01", "label": "normal"}
        ]

        self.current_index = 0
        self.current_email = None
        self.done = False
        self.correct = 0
        self.stage = "classify"
        self.last_classification = None
        self.history = []

    def reset(self):
        self.current_index = 0
        self.done = False
        self.correct = 0
        self.stage = "classify"
        self.history = []
        self.current_email = self.emails[self.current_index]
        return Result(self.state(), 0, False)

    def parse_action(self, action_str):
        if "'1'" in action_str:
            return "mark_spam"
        elif "'2'" in action_str:
            return "mark_important"
        elif "'3'" in action_str:
            return "mark_normal"
        elif "'4'" in action_str:
            return "archive"
        elif "'5'" in action_str:
            return "reply"
        elif "'6'" in action_str:
            return "ignore"
        return "invalid"

    def generate_explanation(self, email, action):
        subject = email["subject"].lower()
        if "win" in subject or "offer" in subject:
            return "Detected promotional keywords → spam"
        elif "urgent" in subject or "account" in subject:
            return "Detected urgency/security keywords → important"
        elif "meeting" in subject:
            return "General communication → normal"
        return f"Action {action} based on learned patterns"

    def step(self, action_str):
        if self.done:
            return Result(self.state(), 0, True)

        action = self.parse_action(action_str)
        correct_label = self.current_email["label"]

        if self.stage == "classify":
            if action == f"mark_{correct_label}":
                reward = 1
                confidence = 0.9
                self.correct += 1
            else:
                reward = -0.5
                confidence = 0.3

            explanation = self.generate_explanation(self.current_email, action)

            self.last_classification = action
            self.stage = "decision"

            return Result(self.state(), reward, False, {
                "confidence": confidence,
                "explanation": explanation
            })

        elif self.stage == "decision":
            if correct_label == "spam" and action == "archive":
                reward = 1
            elif correct_label == "important" and action == "reply":
                reward = 1
            elif correct_label == "normal" and action == "ignore":
                reward = 1
            else:
                reward = -0.5

            explanation = f"Decision {action} based on classification {self.last_classification}"

            self.history.append(self.current_email)

            self.current_index += 1
            self.stage = "classify"

            if self.current_index >= len(self.emails):
                self.done = True
            else:
                self.current_email = self.emails[self.current_index]

            return Result(self.state(), reward, self.done, {
                "explanation": explanation
            })

    def state(self):
        return Observation(self.current_email, self.current_index, self.stage, self.history)

    def get_score(self):
        return self.correct / len(self.emails)
