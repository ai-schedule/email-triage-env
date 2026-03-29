from env import EmailEnv

env = EmailEnv(difficulty="easy")

result = env.reset()
obs = result.observation

print("Goal:", obs.goal)
print("Email:", obs.email)

done = False

while not done:
    print("\nCurrent Email:", obs.email)

    subject = obs.email["subject"].lower()

    # Smarter agent
    if "win" in subject or "offer" in subject or "congratulations" in subject:
        action = "click('1')"  # spam
    elif "meeting" in subject or "project" in subject or "payment" in subject:
        action = "click('2')"  # important
    else:
        action = "click('3')"  # normal

    print("Action:", action)

    result = env.step(action)
    obs = result.observation

    print("Reward:", result.reward)
    print("Done:", result.done)

    done = result.done

print("\nFinal Score:", env.get_score())
