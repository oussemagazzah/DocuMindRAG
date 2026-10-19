"""LLM-as-judge faithfulness, used as a second opinion next to the heuristic score.

Follows the Ragas faithfulness recipe: split the answer into atomic claims,
check each against the retrieved context, score = supported / total claims.
The judge sees only the retrieved context (not the PDFs or the gold answer),
so it measures faithfulness to what the generator was given, not correctness.
"""

import json

from evaluation.llm_client import GeminiClient

JUDGE_SYSTEM = """You are a strict fact-checker for a retrieval-augmented QA system.
You receive a question, the context passages the system retrieved, and the system's answer.

1. Split the answer into short, atomic factual claims. Ignore page citations like [p.3] and
   ignore statements that information is missing or unavailable.
2. For each claim decide whether the context explicitly supports it. A claim that needs
   outside knowledge, or contradicts the context, is NOT supported.
3. Decide whether the answer is a refusal, i.e. it mainly says the documents do not
   contain the answer.

Return only JSON:
{"is_refusal": true|false,
 "claims": [{"claim": "...", "supported": true|false, "reason": "short reason"}]}"""


def judge_prompt(question: str, context: str, answer: str) -> str:
    return f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"


def parse_judgement(text: str) -> dict:
    data = json.loads(text)
    claims = [c for c in data.get("claims", []) if isinstance(c, dict) and "supported" in c]
    supported = sum(1 for c in claims if c["supported"] is True)
    return {
        "judge_is_refusal": bool(data.get("is_refusal", False)),
        "judge_claims": len(claims),
        "judge_supported": supported,
        "judge_faithfulness": round(supported / len(claims), 4) if claims else None,
        "judge_unsupported": [c.get("claim", "") for c in claims if c["supported"] is not True],
    }


def judge_answer(client: GeminiClient, model: str, question: str, context: str, answer: str) -> dict:
    result = client.generate(model, JUDGE_SYSTEM, judge_prompt(question, context, answer),
                             temperature=0.0, max_output_tokens=4096, json_output=True)
    try:
        return parse_judgement(result.text) | {"judge_finish_reason": result.finish_reason}
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        return {"judge_error": f"unparseable judge output: {type(e).__name__}", "judge_faithfulness": None,
                "judge_finish_reason": result.finish_reason}
