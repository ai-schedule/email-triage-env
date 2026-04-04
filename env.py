class Observation:
    def __init__(self, email, step, stage):
        self.goal = "Classify and take action on the email"
        self.url = f"inbox://email/{step}"
        self.screenshot = None
        self.last_action_error = False

        # dynamic actions based on stage
        if stage == "classify":
            elements = {
                "1": {"clickable": True},
                "2": {"clickable": True},
                "3": {"clickable": True},
            }
        else:
            elements = {
                "4": {"clickable": True},
                "5": {"clickable": True},
                "6": {"clickable": True},
            }

        self.metadata = {
            "browsergym_obs": {
                "extra_element_properties": elements
            }
        }

        self.email = {
            "subject": email["subject"],
            "body": email["body"],
            "sender": email["sender"],
            "timestamp": email["timestamp"]
        }

        # priority score
        urgent_keywords = ["urgent", "account", "payment", "alert"]
        score = sum(word in email["subject"].lower() for word in urgent_keywords)
        self.priority_score = min(score / 4, 1.0)

        self.stage = stage


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
            {
                "subject": "URGENT: Account Suspended",
                "body": "Verify immediately",
                "sender": "bank@secure.com",
                "timestamp": "2026-04-01",
                "label": "important"
            },
            {
                "subject": "Win iPhone Now!!!",
                "body": "Click to claim",
                "sender": "spam@promo.com",
                "timestamp": "2026-04-01",
                "label": "spam"
            },
            {
                "subject": "Team Meeting",
                "body": "Tomorrow 10AM",
                "sender": "manager@company.com",
                "timestamp": "2026-04-01",
                "label": "normal"
            }
        ]

        self.current_index = 0
        self.current_email = None
        self.done = False
        self.correct = 0
        self.stage = "classify"
        self.last_classification = None

    def reset(self):
        self.current_index = 0
        self.done = False
        self.correct = 0
        self.stage = "classify"
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

    def step(self, action_str):
        if self.done:
            return Result(self.state(), 0, True)

        action = self.parse_action(action_str)
        correct_label = self.current_email["label"]

        # 🔥 STAGE 1: CLASSIFICATION
        if self.stage == "classify":

            if action == f"mark_{correct_label}":
                reward = 1
                self.correct += 1
            else:
                reward = -0.5

            self.last_classification = action
            self.stage = "decision"

            return Result(self.state(), reward, False)

        # 🔥 STAGE 2: DECISION
        elif self.stage == "decision":

            # correct behavior logic
            if correct_label == "spam" and action == "archive":
                reward = 1
            elif correct_label == "important" and action == "reply":
                reward = 1
            elif correct_label == "normal" and action == "ignore":
                reward = 1
            else:
                reward = -0.5

            # move to next email
            self.current_index += 1
            self.stage = "classify"

            if self.current_index >= len(self.emails):
                self.done = True
            else:
                self.current_email = self.emails[self.current_index]

            return Result(self.state(), reward, self.done)

    def state(self):
        return Observation(self.current_email, self.current_index, self.stage)

    def get_score(self):
        return self.correct / len(self.emails)
