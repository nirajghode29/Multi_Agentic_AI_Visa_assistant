# Multi_Agentic_AI_Visa_assistant
Multi Agent Visa Chatbot assistant using OpenAI and Anthropic API

# 🧠 Agentic Immigration Assistant

A multi-agent AI system that reasons step-by-step through U.S. immigration questions — powered by **GPT-4.1** as the thinker and **Claude** as the validator.

---

## ✨ How It Works

```
User Query → 💭 GPT-4.1 Thinks → 🔎 Claude Validates → 💭 GPT-4.1 Thinks → ... → 🤖 Final Answer
```

1. **💭 Think** — GPT-4.1 breaks the question into discrete reasoning steps
2. **🔎 Validate** — Claude reviews each step for logic and accuracy
3. **🤖 Result** — A structured, well-reasoned final answer with disclaimers

---

## 🗂️ Visa Coverage

| Category | Visas |
|----------|-------|
| 🎓 Student | F-1, OPT, STEM OPT |
| 💼 Employment | H-1B, L-1, TN, O-1, E-3 |
| 🌿 Permanent Residence | EB-1A, EB-1B, EB-2 NIW, I-485 |

---

## 🚀 Quickstart

```bash
pip install openai anthropic python-dotenv
```

Add your keys to `.env`:
```
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

Run:
```bash
persona.py
```

---

## 🛠️ Stack

- `openai` — GPT-4.1 (reasoner)
- `anthropic` — Claude (validator)
- `python-dotenv` — environment config

---

## ⚠️ Disclaimer

> This tool provides **general educational information only** — not legal advice.
> Always consult a licensed immigration attorney for your specific case.

---
