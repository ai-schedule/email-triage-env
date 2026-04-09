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
    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty
        self.load_emails()

        self.current_index = 0
        self.current_email = None
        self.done = False
        self.correct = 0
        self.stage = "classify"
        self.last_classification = None
        self.history = []

    def load_emails(self):
        if self.difficulty == "easy":
            self.emails = [
                {"subject": "Win money now!!!", "body": "Claim prize", "label": "spam"},
                {"subject": "Meeting at 10", "body": "Office meeting", "label": "important"},
                {"subject": "Hello", "body": "How are you?", "label": "normal"}
            ]

        elif self.difficulty == "medium":
            self.emails = [
                {"subject": "Account alert", "body": "Login attempt", "label": "important"},
                {"subject": "Limited time offer", "body": "Buy now", "label": "spam"},
                {"subject": "Project update", "body": "Work progress", "label": "normal"}
            ]

        else:
            self.emails = [
                {"subject": "Re: meeting", "body": "Reschedule discussion", "label": "important"},
                {"subject": "Congratulations!!!", "body": "You are selected", "label": "spam"},
                {"subject": "Newsletter", "body": "Monthly updates", "label": "normal"}
            ]

    def reset(self, task="easy"):
        self.difficulty = task
        self.load_emails()

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

    def step(self, action_str):
        if self.done:
            return Result(self.state(), 0, True)

        action = self.parse_action(action_str)
        correct_label = self.current_email["label"]

        if self.stage == "classify":
            if action == f"mark_{correct_label}":
                reward = 1
                self.correct += 1
            else:
                reward = -0.5

            self.last_classification = action
            self.stage = "decision"

            return Result(self.state(), reward, False)

        else:
            if correct_label == "spam" and action == "archive":
                reward = 1
            elif correct_label == "important" and action == "reply":
                reward = 1
            elif correct_label == "normal" and action == "ignore":
                reward = 1
            else:
                reward = -0.5

            self.history.append(self.current_email)

            self.current_index += 1
            self.stage = "classify"

            if self.current_index >= len(self.emails):
                self.done = True
            else:
                self.current_email = self.emails[self.current_index]

            return Result(self.state(), reward, self.done)

    def state(self):
        return Observation(self.current_email, self.current_index, self.stage, self.history)

    def get_score(self):
        return self.correct / len(self.emails)


# ✅ FINAL GRADER (VERY IMPORTANT)
def grade(env):
    return {
        "score": env.get_score()
    }
