import os
import requests                          # built-in, no pip install needed
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title='Reflection Agent - Ollama', version='1.0')

# ── Ollama Config ────────────────────────────────────────────────────────────
# Instead of Groq cloud API, we call local Ollama server
OLLAMA_URL   = 'http://localhost:11434/api/chat'
OLLAMA_MODEL = 'qwen2.5:0.5b'           # lightweight, runs on 4 GB RAM

# ── System Prompts (IDENTICAL to main.py — do not change) ───────────────────
GENERATOR_PROMPT = '''
You are an expert Python developer.
When given a task, write a clean, working Python function.
Include a docstring. Handle edge cases.
Return ONLY the Python code — no explanation, no markdown fences.
'''

CRITIC_PROMPT = '''
You are a senior Python code reviewer.
Evaluate the given code against these five criteria:
1. Correctness   — does it produce the right output?
2. Edge cases    — does it handle None, empty, zero, negatives?
3. Readability   — clear names, comments, easy to follow?
4. Efficiency    — no unnecessary loops or operations?
5. Security      — no eval on user input, no hardcoded secrets?

For each criterion found to have a problem, write:
ISSUE [criterion]: <specific problem and why it matters>

If ALL five criteria are met with no issues, respond with exactly one word:
APPROVED

Never say APPROVED if any criterion has an issue.
Be specific. Generic feedback is useless.
'''

REVISION_PROMPT = '''
You are an expert Python developer revising your previous code.

Your original code:
{original}

Code review critique received:
{critique}

Rewrite the function to fix every issue raised.
Do not just acknowledge the critique — actually fix each problem.
Return ONLY the corrected Python code.
'''

# ── Request / Response Schemas (IDENTICAL to main.py) ───────────────────────
class ReflectRequest(BaseModel):
    task: str
    max_rounds: int = 3

class ReflectResponse(BaseModel):
    final_code:  str
    round_count: int
    approved:    bool
    critiques:   list[str]

# ── Core LLM Call — ONLY THIS FUNCTION CHANGES from main.py ─────────────────
def call_llm(system: str, user: str) -> str:
    '''
    Same interface as the Groq version.
    Calls Ollama local API instead of Groq cloud API.
    The reflection loop calls this function identically in both versions.
    
    Key difference in response shape:
      Groq:   response.choices[0].message.content
      Ollama: response.json()['message']['content']
    '''
    payload = {
        'model':    OLLAMA_MODEL,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': user},
        ],
        'stream':  False,
        'options': {'temperature': 0.3},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()['message']['content'].strip()

# ── Reflection Loop (IDENTICAL to main.py — not changed) ────────────────────
def run_reflection(task: str, max_rounds: int) -> ReflectResponse:
    critiques    = []
    current_code = ''

    # Round 0: Initial generation
    current_code = call_llm(
        system=GENERATOR_PROMPT,
        user=f'Write a Python function for this task:\n{task}'
    )

    for round_num in range(1, max_rounds + 1):

        # Critic evaluates current code
        critique = call_llm(
            system=CRITIC_PROMPT,
            user=f'Review this Python code:\n\n{current_code}'
        )

        # Stopping condition: approved?
        if critique.strip().upper() == 'APPROVED':
            critiques.append('APPROVED')
            return ReflectResponse(
                final_code  = current_code,
                round_count = round_num,
                approved    = True,
                critiques   = critiques,
            )

        # Issues found — revise
        critiques.append(critique)
        revision_message = REVISION_PROMPT.format(
            original=current_code,
            critique=critique,
        )
        current_code = call_llm(
            system=GENERATOR_PROMPT,
            user=revision_message,
        )

    # Max rounds reached without approval
    return ReflectResponse(
        final_code  = current_code,
        round_count = max_rounds,
        approved    = False,
        critiques   = critiques,
    )

# ── Endpoints (IDENTICAL to main.py) ────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok', 'model': OLLAMA_MODEL}   # shows qwen2.5:0.5b

@app.post('/reflect', response_model=ReflectResponse)
def reflect(req: ReflectRequest):
    if not req.task.strip():
        raise HTTPException(status_code=400, detail='task field cannot be empty')
    return run_reflection(req.task, req.max_rounds)