import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

load_dotenv()  # Loads GROQ_API_KEY from .env file

app = FastAPI(title='Reflection Agent', version='1.0')

# ── LLM Client ──────────────────────────────────────────────────────────────
# Groq reads the API key from the environment automatically
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL  = 'llama-3.1-8b-instant'

# ── System Prompts (Role Definitions) ───────────────────────────────────────
# SAME model, DIFFERENT roles — just different system prompts

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

# ── Request / Response Schemas ───────────────────────────────────────────────
class ReflectRequest(BaseModel):
    task: str           # e.g. 'Write a function that finds duplicate items in a list'
    max_rounds: int = 3 # maximum reflection iterations (safety cap)

class ReflectResponse(BaseModel):
    final_code:  str        # the approved or best final function
    round_count: int        # how many rounds the loop ran
    approved:    bool       # did the critic approve before max_rounds?
    critiques:   list[str]  # critique from each round

# ── Core LLM Call ────────────────────────────────────────────────────────────
def call_llm(system: str, user: str) -> str:
    '''Single LLM call — takes a system prompt and a user message.'''
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': user},
        ],
        temperature=0.3,   # Low = more consistent, deterministic code output
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()

# ── Reflection Loop ──────────────────────────────────────────────────────────
def run_reflection(task: str, max_rounds: int) -> ReflectResponse:
    '''
    Implements the Reflection Pattern:
    1. Generator writes initial code
    2. Critic evaluates it against the checklist
    3. If APPROVED → return result
    4. If issues found → pass critique back to Generator
    5. Repeat up to max_rounds
    '''
    critiques    = []
    current_code = ''

    # Round 0: Initial generation (no critique yet)
    current_code = call_llm(
        system=GENERATOR_PROMPT,
        user=f'Write a Python function for this task:\n{task}'
    )

    for round_num in range(1, max_rounds + 1):

        # Critic evaluates the current code
        critique = call_llm(
            system=CRITIC_PROMPT,
            user=f'Review this Python code:\n\n{current_code}'
        )

        # Check stopping condition: did critic approve?
        if critique.strip().upper() == 'APPROVED':
            critiques.append('APPROVED')
            return ReflectResponse(
                final_code  = current_code,
                round_count = round_num,
                approved    = True,
                critiques   = critiques,
            )

        # Critic found issues — record them and request revision
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

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL}

@app.post('/reflect', response_model=ReflectResponse)
def reflect(req: ReflectRequest):
    if not req.task.strip():
        raise HTTPException(status_code=400, detail='task field cannot be empty')
    return run_reflection(req.task, req.max_rounds)