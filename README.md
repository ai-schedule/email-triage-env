# 📧 Email Triage RL Environment

## 📌 Overview
This project implements a real-world reinforcement learning environment where an AI agent learns to classify emails into categories:
- Spam
- Important
- Normal

The environment follows the OpenEnv API with `reset()`, `step()`, and `state()` functions.

---

## 🎯 Objective
Train an AI agent to correctly classify incoming emails and maximize reward.

---

## 🧠 Environment Design

### 🔹 State (Observation)
Each state contains:
- Email subject
- Email body
- Classification goal
- Metadata for actions

Example:
```python
{
  "subject": "Win money now!!!",
  "body": "Claim prize",
  "label": "spam"
}
