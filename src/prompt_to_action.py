"""Convert natural language prompts to robot action sequences.

Uses Claude API when available, falls back to keyword matching.
"""

import json
import os
import re
from typing import Dict, List, Optional

from src.robot_sim import ACTION_REGISTRY, get_available_actions


# ── Keyword-based fallback parser ────────────────────────────────────────────

KEYWORD_MAP = {
    "walk forward": [{"action": "walk_forward", "duration": 3.0}],
    "walk": [{"action": "walk_forward", "duration": 3.0}],
    "move forward": [{"action": "walk_forward", "duration": 3.0}],
    "go forward": [{"action": "walk_forward", "duration": 3.0}],
    "walk backward": [{"action": "walk_backward", "duration": 3.0}],
    "go back": [{"action": "walk_backward", "duration": 3.0}],
    "back up": [{"action": "walk_backward", "duration": 2.0}],
    "run": [{"action": "run", "duration": 3.0}],
    "sprint": [{"action": "run", "duration": 3.0}],
    "fast": [{"action": "run", "duration": 3.0}],
    "turn left": [{"action": "turn_left", "duration": 2.0}],
    "go left": [{"action": "turn_left", "duration": 2.0}],
    "turn right": [{"action": "turn_right", "duration": 2.0}],
    "go right": [{"action": "turn_right", "duration": 2.0}],
    "wave": [{"action": "wave", "duration": 2.5}],
    "wave hello": [{"action": "wave", "duration": 2.5}],
    "say hello": [{"action": "wave", "duration": 2.5}],
    "hello": [{"action": "wave", "duration": 2.5}],
    "hi": [{"action": "wave", "duration": 2.5}],
    "greet": [{"action": "wave", "duration": 2.5}],
    "bow": [{"action": "bow", "duration": 2.5}],
    "squat": [{"action": "squat", "duration": 2.5}],
    "crouch": [{"action": "squat", "duration": 2.5}],
    "kick": [{"action": "kick", "duration": 2.0}],
    "raise arms": [{"action": "raise_arms", "duration": 2.0}],
    "hands up": [{"action": "raise_arms", "duration": 2.0}],
    "arms up": [{"action": "raise_arms", "duration": 2.0}],
    "celebrate": [{"action": "raise_arms", "duration": 2.0}],
    "sit": [{"action": "sit", "duration": 2.0}],
    "sit down": [{"action": "sit", "duration": 2.0}],
    "stand": [{"action": "stand", "duration": 1.5}],
    "stand up": [{"action": "stand", "duration": 1.5}],
    "stop": [{"action": "stand", "duration": 1.5}],
    "do a lap": [
        {"action": "walk_forward", "duration": 2.0},
        {"action": "turn_left", "duration": 2.0},
        {"action": "walk_forward", "duration": 2.0},
        {"action": "turn_left", "duration": 2.0},
    ],
    "show off": [
        {"action": "wave", "duration": 2.0},
        {"action": "walk_forward", "duration": 2.0},
        {"action": "kick", "duration": 2.0},
        {"action": "dance", "duration": 3.0},
        {"action": "bow", "duration": 2.0},
    ],
    "show me": [
        {"action": "wave", "duration": 2.0},
        {"action": "walk_forward", "duration": 2.0},
        {"action": "kick", "duration": 2.0},
        {"action": "dance", "duration": 3.0},
        {"action": "bow", "duration": 2.0},
    ],
    "what you can do": [
        {"action": "wave", "duration": 2.0},
        {"action": "walk_forward", "duration": 2.0},
        {"action": "kick", "duration": 2.0},
        {"action": "dance", "duration": 3.0},
        {"action": "bow", "duration": 2.0},
    ],
    "dance": [{"action": "dance", "duration": 4.0}],
    "boogie": [{"action": "dance", "duration": 4.0}],
    "patrol": [
        {"action": "walk_forward", "duration": 3.0},
        {"action": "turn_right", "duration": 2.0},
        {"action": "walk_forward", "duration": 3.0},
        {"action": "turn_right", "duration": 2.0},
    ],
    "explore": [
        {"action": "walk_forward", "duration": 2.0},
        {"action": "turn_left", "duration": 1.5},
        {"action": "walk_forward", "duration": 2.0},
        {"action": "turn_right", "duration": 1.5},
        {"action": "walk_forward", "duration": 2.0},
    ],
}

# Default if nothing matches
DEFAULT_SEQUENCE = [
    {"action": "stand", "duration": 1.0},
    {"action": "walk_forward", "duration": 3.0},
]


def _parse_duration(prompt: str) -> Optional[float]:
    """Extract a duration hint from the prompt (e.g., '5 steps', '3 seconds')."""
    m = re.search(r'(\d+)\s*(steps?|seconds?|sec|s)\b', prompt.lower())
    if m:
        n = int(m.group(1))
        if 'step' in m.group(2):
            return n * 0.8  # each "step" ~ 0.8s
        return float(n)
    return None


def parse_prompt_keywords(prompt: str) -> List[Dict]:
    """Parse prompt using keyword matching. Returns action sequence."""
    prompt_lower = prompt.lower().strip()
    duration_hint = _parse_duration(prompt)

    # Try longest keyword matches first
    sorted_keys = sorted(KEYWORD_MAP.keys(), key=len, reverse=True)
    for keyword in sorted_keys:
        if keyword in prompt_lower:
            sequence = [dict(a) for a in KEYWORD_MAP[keyword]]  # deep copy
            if duration_hint:
                # Apply duration hint to first action
                sequence[0]["duration"] = min(duration_hint, 10.0)
            return sequence

    return list(DEFAULT_SEQUENCE)


# ── Claude API parser ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a robot action planner for a Unitree Go2 quadruped robot.
Given a natural language instruction, output a JSON array of action steps.

Available actions:
{actions}

Each step is an object with:
- "action": one of the action names above
- "duration": seconds (0.5 to 10.0)

Rules:
- Always start with a brief "stand" if the instruction implies starting from rest
- Keep total duration under 15 seconds
- Be creative with combinations for complex instructions
- Output ONLY the JSON array, no other text
"""


def _call_llm(prompt: str, system: str) -> Optional[str]:
    """Call an LLM (Anthropic or OpenAI) and return the response text."""
    # Try Anthropic first
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic API failed: {e}")

    # Try OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API failed: {e}")

    return None


def parse_prompt_llm(prompt: str) -> List[Dict]:
    """Parse prompt using LLM (Anthropic or OpenAI). Falls back to keywords."""
    actions_desc = "\n".join(
        f'- "{name}": {desc}'
        for name, desc in get_available_actions().items()
    )
    system = SYSTEM_PROMPT.format(actions=actions_desc)

    text = _call_llm(prompt, system)
    if text is None:
        return parse_prompt_keywords(prompt)

    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            sequence = json.loads(match.group())
        else:
            sequence = json.loads(text)

        validated = []
        for step in sequence:
            if isinstance(step, dict) and "action" in step:
                action = step["action"]
                if action in ACTION_REGISTRY:
                    duration = float(step.get("duration", ACTION_REGISTRY[action]["duration"]))
                    duration = max(0.5, min(duration, 10.0))
                    validated.append({"action": action, "duration": duration})

        if validated:
            return validated
    except Exception as e:
        print(f"LLM response parsing failed: {e}")

    return parse_prompt_keywords(prompt)


def parse_prompt(prompt: str) -> List[Dict]:
    """Parse a natural language prompt into an action sequence.

    Uses LLM if ANTHROPIC_API_KEY or OPENAI_API_KEY is set, otherwise keywords.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        sequence = parse_prompt_llm(prompt)
    else:
        sequence = parse_prompt_keywords(prompt)

    # Cap total duration at 15s for responsiveness
    total = 0
    capped = []
    for step in sequence:
        dur = step.get("duration", 2.0)
        if total + dur > 15.0:
            remaining = 15.0 - total
            if remaining > 0.5:
                capped.append({**step, "duration": remaining})
            break
        capped.append(step)
        total += dur
    return capped or sequence[:3]


def describe_sequence(sequence: List[Dict]) -> str:
    """Generate a human-readable description of an action sequence."""
    parts = []
    total_duration = 0
    for step in sequence:
        name = step["action"]
        dur = step.get("duration", ACTION_REGISTRY.get(name, {}).get("duration", 1.0))
        desc = ACTION_REGISTRY.get(name, {}).get("description", name)
        parts.append(f"  {desc} ({dur:.1f}s)")
        total_duration += dur

    header = f"Action plan ({total_duration:.1f}s total):"
    return header + "\n" + "\n".join(parts)


if __name__ == "__main__":
    # Test various prompts
    test_prompts = [
        "Walk forward",
        "Turn left and then walk",
        "Do a little dance",
        "Run fast for 5 seconds",
        "Sit down",
        "Show me what you can do",
        "Patrol the area",
    ]

    for prompt in test_prompts:
        seq = parse_prompt(prompt)
        print(f"\nPrompt: '{prompt}'")
        print(describe_sequence(seq))
