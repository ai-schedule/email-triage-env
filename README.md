---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
---

# Email Triage RL Environment

## Overview
Multi-step RL environment for email classification and decision making.

## Advanced Features
- Multi-step workflow (classify → decide)
- Confidence-based reward
- Priority scoring
- Explainable AI
- Memory (history tracking)

## Design Philosophy
Simulates real-world email systems with sequential decisions and context awareness.

## API
POST /reset  
POST /step  

Actions:
- click('1') spam
- click('2') important
- click('3') normal
- click('4') archive
- click('5') reply
- click('6') ignore
