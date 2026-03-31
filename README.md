---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Email Triage RL Environment

This is a reinforcement learning environment where an agent learns to classify emails.

## API

POST /reset  
POST /step  

Actions:
- click('1') → spam  
- click('2') → important  
- click('3') → normal  
