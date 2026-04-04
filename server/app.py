from fastapi import FastAPI
from env import EmailEnv

app = FastAPI()
env = EmailEnv()

@app.post("/reset")
def reset():
    result = env.reset()
    obs = result.observation

    return {
        "observation": {
            "goal": obs.goal,
            "email": obs.email,
            "stage": obs.stage,
            "priority_score": obs.priority_score,
            "history": obs.history
        },
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

@app.post("/step")
def step(action: str):
    result = env.step(action)
    obs = result.observation

    return {
        "observation": {
            "goal": obs.goal,
            "email": obs.email,
            "stage": obs.stage,
            "priority_score": obs.priority_score,
            "history": obs.history
        },
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

@app.get("/")
def home():
    return {"status": "running"}


def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
