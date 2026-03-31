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

        self.email = email


class Result:
    def __init__(self, observation, reward, done):
        self.observation = observation
        self.reward = reward
        self.done = done


class EmailEnv:
    def __init__(self, difficulty="easy"):
        self.valid_actions = ["mark_spam", "mark_important", "mark_normal"]

        self.emails = [
            {"subject": "Win money now!!!", "body": "Claim prize", "label": "spam"},
            {"subject": "Meeting at 10", "body": "Office meeting", "label": "important"},
            {"subject": "Hello", "body": "How are you?", "label": "normal"},
        ]

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

        if action == f"mark_{correct_label}":
            reward = 1
            self.correct += 1
        elif action == "invalid":
            reward = -1
        else:
            reward = -0.5

        self.current_index += 1

        if self.current_index >= len(self.emails):
            self.done = True
        else:
            self.current_email = self.emails[self.current_index]

        return Result(self.state(), reward, self.done)

    def state(self):
        return Observation(self.current_email, self.current_index)

    def get_score(self):
        return self.correct / len(self.emails)
