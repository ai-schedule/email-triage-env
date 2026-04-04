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

        # ✅ UPDATED: richer email info
        self.email = {
            "subject": email["subject"],
            "body": email["body"],
            "sender": email["sender"],
            "timestamp": email["timestamp"]
        }


class Result:
    def __init__(self, observation, reward, done):
        self.observation = observation
        self.reward = reward
        self.done = done


class EmailEnv:
    def __init__(self, difficulty="easy"):
        self.valid_actions = ["mark_spam", "mark_important", "mark_normal"]

        # ✅ REALISTIC EMAIL DATASET
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

        self.medium_emails = [
            {
                "subject": "Invoice Pending Payment",
                "body": "Please clear dues before deadline",
                "sender": "billing@service.com",
                "timestamp": "2026-04-02 09:00",
                "label": "important"
            },
            {
                "subject": "Limited Time Offer!!!",
                "body": "Huge discounts available",
                "sender": "ads@promo.com",
                "timestamp": "2026-04-02 10:00",
                "label": "spam"
            },
            {
                "subject": "Weekend Plan",
                "body": "Shall we go out this weekend?",
                "sender": "friend@gmail.com",
                "timestamp": "2026-04-02 12:00",
                "label": "normal"
            }
        ]

        self.hard_emails = [
            {
                "subject": "Payment Failed Alert",
                "body": "If not you, contact support",
                "sender": "noreply@bank.com",
                "timestamp": "2026-04-03 08:00",
                "label": "important"
            },
            {
                "subject": "Congratulations! You are selected",
                "body": "Claim reward now",
                "sender": "lottery@unknown.com",
                "timestamp": "2026-04-03 09:00",
                "label": "spam"
            },
            {
                "subject": "Reminder: Submit Assignment",
                "body": "Deadline tonight",
                "sender": "professor@college.edu",
                "timestamp": "2026-04-03 10:00",
                "label": "important"
            }
        ]

        # difficulty selection
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
