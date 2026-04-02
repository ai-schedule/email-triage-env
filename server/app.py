from fastapi import FastAPI
from env import EmailEnv

app = FastAPI()
env = EmailEnv()

@app.post("/reset")
def reset():
    result = env.reset()
    return {
        "observation": {
            "goal": result.observation.goal,
            "email": result.observation.email
        },
        "reward": result.reward,
        "done": result.done
    }

@app.post("/step")
def step(action: str):
    result = env.step(action)
    return {
        "observation": {
            "goal": result.observation.goal,
            "email": result.observation.email
        },
        "reward": result.reward,
        "done": result.done
    }

@app.get("/")
def home():
    return {"status": "running"}


# ✅ REQUIRED MAIN FUNCTION
def main():
    return app


# ✅ THIS WAS MISSING (CRITICAL FIX)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
