#!/usr/bin/env python
"""
generate_fig4_feedback.py -- verbalizes the three items of Fig. 4 with GPT-4o-mini
(temperature 0.2), following the prompt of Generate_Feedback.ipynb with the
level-specific instructions of Section V-E, and writes fig4_feedback_text.txt.

    export OPENAI_API_KEY=...        # or set API_KEY below
    python generate_fig4_feedback.py
    python make_figures.py           # re-renders Fig. 4 with the model output

Python 3.6 compatible; needs only requests.
"""
import os
import json
import requests

API_KEY = os.environ.get("OPENAI_API_KEY", "")
URL = "https://api.openai.com/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))

COMMON = """### ROLE
You are an Eminent Piano Professor conducting a high-level Masterclass.
Your objective is to provide a brief, authoritative 'Pedagogical Annotation' that corrects the student's execution by bridging the gap between the score's intent and their actual performance.

### PERFORMANCE DATA
1. Segment: Measures {segment}
2. Score Requirement (MusicXML): {score}
3. Performance Discrepancy:
{discrepancy}

### LEVEL OF THE REMARK
{level_rule}

### ANNOTATION PRINCIPLES
1. COMMENCEMENT: Start strictly with "Measures {segment}: ".
2. ARTISTIC DIAGNOSIS: Directly compare the score's markings with the student's actual sound.
3. MASTER'S ADVICE: Provide a single, potent piece of technical or interpretive advice.
4. STYLE: Concise (at most three sentences), dignified, and profoundly musical. DO NOT use numbers, weights, or technical jargon like "L1/L2" or "guideline."
5. LANGUAGE: English.
"""

L1_RULE = ("This remark concerns the category '{category}' as a whole. Several of its aspects deviate "
           "together and none dominates, so name the category, make clear that more than one aspect "
           "contributes, and propose a way of practising the area as a whole. Do not single out one aspect.")
L2_RULE = ("This remark concerns one specific aspect, '{name}' ({description}). Name it through its two "
           "opposing terms and state the direction of the change ({change}).")


def anchors(name):
    a, b = [t.strip() for t in name.split(" - ", 1)]
    return a, b


def tendency(d):
    """Direction in words: where the student is, where experts expect, which way to correct."""
    a, b = anchors(d["name"])
    target = b if d["expert_norm"] > d["performance"] else a
    other = a if target == b else b
    return ("'%s': the student plays closer to '%s' than experts expect (student %.2f, expert norm %.2f "
            "on a scale from -1 = '%s' to 1 = '%s'); the correction is toward '%s'."
            % (d["name"], other, d["performance"], d["expert_norm"], a, b, target))


def prompt(item):
    if item["level"] == "L1":
        lines = ["   - " + tendency(d) for d in item["descriptors"]]
        rule = L1_RULE.format(category=item["category"])
    else:
        d = item["descriptor"]
        lines = ["   - " + tendency(d)]
        rule = L2_RULE.format(**d)
    return COMMON.format(segment=item["segment"], score=json.dumps(item["score_context"]),
                         discrepancy="\n".join(lines), level_rule=rule)


def call(text):
    data = {"model": "gpt-4o-mini", "temperature": 0.2,
            "messages": [{"role": "system", "content": "You are a professional music performance assistant specializing in causal-based interpretable feedback."},
                         {"role": "user", "content": text}]}
    r = requests.post(URL, headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY},
                      data=json.dumps(data), timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    if not API_KEY:
        raise SystemExit("set OPENAI_API_KEY")
    spec = json.load(open(os.path.join(HERE, "fig4_llm_inputs.json"), encoding="utf-8"))
    out = ["# GPT-4o-mini output (temperature 0.2) for the items of fig4_llm_inputs.json",
           "# Format: heading | text"]
    for item in spec["items"]:
        text = call(prompt(item))
        head = "Measures %s (%s):" % (item["segment"].replace("-", "–"),
                                     item["category"] if item["level"] == "L1" else item["descriptor"]["name"].replace(" - ", " – "))
        body = text.split(":", 1)[1].strip() if text.startswith("Measures") and ":" in text else text
        out.append("%s | %s" % (head, body.replace("\n", " ")))
        print(head, body, "\n")
    open(os.path.join(HERE, "fig4_feedback_text.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
