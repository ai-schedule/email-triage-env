# -------- Observation Class --------
class Observation:
    def __init__(self, email, step):
        self.goal = "Classify the email correctly"
        self.url = f"inbox://email/{step}"
        self.screenshot = None
        self.last_action_error = False

        # Clickable elements (BIDs)
        self.metadata = {
            "browsergym_obs": {
                "extra_element_properties": {
                    "1": {"clickable": True, "label": "mark_spam"},
                    "2": {"clickable": True, "label": "mark_important"},
                    "3": {"clickable": True, "label": "mark_normal"},
                }
            }
        }

        self.email = email


# -------- Result Class --------
class Result:
    def __init__(self, observation, reward, done):
        self.observation = observation
        self.reward = reward
        self.done = done


# -------- Main Environment --------
class EmailEnv:
    def __init__(self, difficulty="easy"):

        # -------- Valid Actions --------
        self.valid_actions = ["mark_spam", "mark_important", "mark_normal"]

        # -------- Email Dataset --------
        self.easy_emails = [
            {"subject": "Win money now!!!", "body": "Claim prize", "label": "spam"},
            {"subject": "Meeting at 10", "body": "Office meeting", "label": "important"},
            {"subject": "Hello", "body": "How are you?", "label": "normal"},
        ]

        self.medium_emails = [
            {"subject": "Limited Offer", "body": "Buy now", "label": "spam"},
            {"subject": "Project Update", "body": "Deadline tomorrow", "label": "important"},
            {"subject": "Check this out", "body": "Interesting article", "label": "normal"},
        ]

        self.hard_emails = [
            {"subject": "Payment Failed", "body": "Contact support if not you", "label": "important"},
            {"subject": "Congratulations!!!", "body": "You won lottery", "label": "spam"},
            {"subject": "Reminder", "body": "Don't forget to review", "label": "important"},
        ]

        # -------- Difficulty Selection --------
        if difficulty == "easy":
            self.emails = self.easy_emails
        elif difficulty == "medium":
            self.emails = self.medium_emails
        else:
            self.emails = self.hard_emails

        self.current_index = 0
        self.current_email = None
        self.done = False
        self.correct = 0

    # -------- Reset --------
    def reset(self):
        self.current_index = 0
        self.done = False
        self.correct = 0

        self.current_email = self.emails[self.current_index]

        return Result(self.state(), 0, False)

    # -------- Improved Action Parser --------
    def parse_action(self, action_str):
        action_str = action_str.lower().strip()

        if "click" in action_str:
            if "'1'" in action_str:
                return "mark_spam"
            elif "'2'" in action_str:
                return "mark_important"
            elif "'3'" in action_str:
                return "mark_normal"

        return "invalid"

    # -------- Step Function --------
    def step(self, action_str):
        if self.done:
            return Result(self.state(), 0, True)

        action = self.parse_action(action_str)
        correct_label = self.current_email["label"]

        # -------- Reward Logic --------
        if action == "invalid":
            reward = -1
        elif action == f"mark_{correct_label}":
            reward = 1
            self.correct += 1
        else:
            reward = -0.5

        # Move to next email
        self.current_index += 1

        if self.current_index >= len(self.emails):
            self.done = True
        else:
            self.current_email = self.emails[self.current_index]

        return Result(self.state(), reward, self.done)

    # -------- State --------
    def state(self):
        return Observation(
            email=self.current_email,
            step=self.current_index
        )

    # -------- Grader --------
    def get_score(self):
        return self.correct / len(self.emails)
