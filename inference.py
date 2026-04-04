import os
from openai import OpenAI
from env import EmailEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def main():
    print("START")

    env = EmailEnv()
    result = env.reset()
    obs = result.observation

    step = 1
    done = False

    while not done:
        prompt = f"""
        Email:
        {obs.email}
        Stage: {obs.stage}
        """

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=50
            )
            response = completion.choices[0].message.content.lower()
        except Exception:
            response = "spam"

        if "spam" in response:
            action = "click('1')"
        elif "important" in response:
            action = "click('2')"
        elif "normal" in response:
            action = "click('3')"
        elif "archive" in response:
            action = "click('4')"
        elif "reply" in response:
            action = "click('5')"
        else:
            action = "click('6')"

        print(f"STEP {step}: {action}")

        result = env.step(action)
        obs = result.observation
        done = result.done
        step += 1

    print("END")


if __name__ == "__main__":
    main()
