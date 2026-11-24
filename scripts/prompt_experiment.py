#!/usr/bin/env python
"""
prompt_experiment.py -- verbalizes every selected item of case_items.json with
GPT-4o-mini under two prompt variants and checks the direction of each remark.

  numeric : deviations given as numbers on the descriptor scale (as in the original
            Generate_Feedback.ipynb prompt)
  verbal  : the same numbers plus the direction in words ("the student plays closer
            to A than experts expect; the correction is toward B")

Both variants share everything else: role, segment, importance of the segment,
score directives (when a score is available), level-specific instructions, and
style constraints. A judge model (gpt-4o, temperature 0) reads each remark and
reports, for every descriptor of the item, which way the remark asks the student
to move. Results are appended to prompt_results.jsonl (resumable).

    OPENAI_API_KEY=... python prompt_experiment.py
    python prompt_experiment.py summary
Python 3.6 compatible; needs requests.
"""
import os
import re
import sys
import json
import time
import requests

API_KEY = os.environ.get("OPENAI_API_KEY", "")
URL = "https://api.openai.com/v1/chat/completions"
GEN_MODEL, JUDGE_MODEL = "gpt-4o-mini", "gpt-4o"
OUT = "prompt_results.jsonl"
ORD = {1: "most", 2: "second most", 3: "third most", 4: "fourth most", 5: "fifth most"}

SYSTEM = "You are a professional music performance assistant specializing in causal-based interpretable feedback."
TEMPLATE = """### ROLE
You are an Eminent Piano Professor conducting a high-level Masterclass.
Your objective is to provide a brief, authoritative 'Pedagogical Annotation' that corrects the student's execution by bridging the gap between the score's intent and their actual performance.

### PERFORMANCE DATA
1. Piece: {piece}
2. Segment: Measures {segment}
3. Why this segment: it is the {importance} heavily weighted passage in the evaluation of this performance.
{score_line}4. Performance Discrepancy:
{discrepancy}

### LEVEL OF THE REMARK
{level_rule}

### ANNOTATION PRINCIPLES
1. COMMENCEMENT: Start strictly with "Measures {segment}: ".
2. IMPORTANCE: Open with a short clause saying why this passage deserves attention.
3. ARTISTIC DIAGNOSIS: Directly compare what the music asks for with the student's actual sound.
4. MASTER'S ADVICE: Provide a single, potent piece of technical or interpretive advice.
5. STYLE: Concise (at most three sentences), dignified, and profoundly musical. DO NOT use numbers, weights, or technical jargon like "L1/L2" or "guideline."
6. LANGUAGE: English.
"""
L2_RULE = ("This remark concerns one specific aspect, '{name}'. Name it through its two opposing terms "
           "and state the direction of the change the student should make.")
L1_RULE = ("This remark concerns the category '{category}' as a whole. Several of its aspects deviate together "
           "and none dominates, so name the category, make clear that more than one aspect contributes, and "
           "propose a way of practising the area as a whole. Do not single out one aspect.")
L0_RULE = ("This remark concerns the passage as a whole. Its weakness is spread over several areas; describe the "
           "overall impression and point to the areas that contribute most.")


def numeric_line(d):
    a, b = d["anchors"]
    return "'%s' (scale from -1 = '%s' to 1 = '%s'): student %.2f, expert norm %.2f" % (
        d["name"], a, b, d["performance"], d["expert_norm"])


def verbal_line(d):
    a, b = d["anchors"]
    other = a if d["toward"] == b else b
    return ("'%s': the student plays closer to '%s' than experts expect (student %.2f, expert norm %.2f "
            "on a scale from -1 = '%s' to 1 = '%s'); the correction is toward '%s'."
            % (d["name"], other, d["performance"], d["expert_norm"], a, b, d["toward"]))


def build_prompt(perf, item, variant):
    seg = "%d-%d" % tuple(item["measures"])
    line = verbal_line if variant == "verbal" else numeric_line
    disc = "\n".join("   - " + line(d) for d in item["deviations"])
    if item["level"] == "L2":
        rule = L2_RULE.format(name=item["node"])
    elif item["level"] == "L1":
        rule = L1_RULE.format(category=item["category"])
    else:
        rule = L0_RULE
    sc = item.get("score_context")
    score_line = "   Score directives (MusicXML): %s\n" % json.dumps(sc) if sc else ""
    return TEMPLATE.format(piece=perf["piece"], segment=seg, importance=ORD.get(item["segment_rank"], "a"),
                           score_line=score_line, discrepancy=disc, level_rule=rule)


def chat(model, messages, temperature, json_mode=False, tries=4):
    data = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": 300}
    if json_mode:
        data["response_format"] = {"type": "json_object"}
    for k in range(tries):
        try:
            r = requests.post(URL, headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY},
                              data=json.dumps(data), timeout=90)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(5 * (k + 1))
                continue
            raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:200]))
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    raise RuntimeError("request failed")


def judge(text, item):
    aspects = [{"aspect": d["name"], "A": d["anchors"][0], "B": d["anchors"][1]} for d in item["deviations"]]
    q = ("Read the piano feedback below. For each listed aspect, an opposing pair of terms A and B, decide which change "
         "the feedback asks the student to make: \"A\" if it asks to move toward A, \"B\" if toward B, or \"none\" if the "
         "aspect is not addressed or the direction is unclear. Judge only what the feedback says.\n\n"
         "FEEDBACK:\n%s\n\nASPECTS:\n%s\n\n"
         "Return JSON: {\"aspects\": [{\"aspect\": <aspect>, \"answer\": \"A\"|\"B\"|\"none\"}]}"
         % (text, json.dumps(aspects, ensure_ascii=False)))
    out = json.loads(chat(JUDGE_MODEL, [{"role": "user", "content": q}], 0.0, json_mode=True))
    ans = {a.get("aspect"): a.get("answer") for a in out.get("aspects", [])}
    res = []
    for d in item["deviations"]:
        v = ans.get(d["name"], "none")
        said = d["anchors"][0] if v == "A" else d["anchors"][1] if v == "B" else None
        res.append({"aspect": d["name"], "said": said, "target": d["toward"],
                    "verdict": "none" if said is None else ("correct" if said == d["toward"] else "reversed")})
    return res


def run():
    if not API_KEY:
        raise SystemExit("set OPENAI_API_KEY")
    D = json.load(open("case_items.json", encoding="utf-8"))
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["pid"], r["rank"], r["variant"]))
    with open(OUT, "a", encoding="utf-8") as fo:
        for perf in D["performances"]:
            for item in perf["items"]:
                for variant in ("numeric", "verbal"):
                    key = (perf["pid"], item["rank"], variant)
                    if key in done:
                        continue
                    p = build_prompt(perf, item, variant)
                    text = chat(GEN_MODEL, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}], 0.2)
                    j = judge(text, item)
                    lo, hi = item["measures"]
                    rec = {"pid": perf["pid"], "rank": item["rank"], "variant": variant, "level": item["level"],
                           "node": item["node"], "measures": item["measures"], "prompt": p, "feedback": text,
                           "judge": j, "mentions_measures": bool(re.search(r"\b%d\b" % lo, text) and re.search(r"\b%d\b" % hi, text))}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fo.flush()
                    print(perf["pid"], item["rank"], variant, [x["verdict"] for x in j])


def summary():
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    for variant in ("numeric", "verbal"):
        R = [r for r in rows if r["variant"] == variant]
        asp = [x for r in R for x in r["judge"]]
        n = len(asp)
        c = sum(x["verdict"] == "correct" for x in asp)
        w = sum(x["verdict"] == "reversed" for x in asp)
        z = sum(x["verdict"] == "none" for x in asp)
        l1 = [r for r in R if r["level"] == "L1"]
        multi = sum(sum(x["verdict"] != "none" for x in r["judge"]) >= 2 for r in l1)
        items_all_ok = sum(all(x["verdict"] == "correct" for x in r["judge"]) for r in R)
        print("%-8s remarks %d | aspects %d: correct %d (%.1f%%), reversed %d (%.1f%%), not addressed %d (%.1f%%) | "
              "remarks with every aspect correct %d (%.1f%%) | L1 remarks naming >=2 aspects %d/%d | measures cited %d/%d"
              % (variant, len(R), n, c, 100.0 * c / n, w, 100.0 * w / n, z, 100.0 * z / n,
                 items_all_ok, 100.0 * items_all_ok / len(R), multi, len(l1), sum(r["mentions_measures"] for r in R), len(R)))
    # paired view per remark
    by = {}
    for r in rows:
        by.setdefault((r["pid"], r["rank"]), {})[r["variant"]] = all(x["verdict"] == "correct" for x in r["judge"])
    pairs = [v for v in by.values() if len(v) == 2]
    print("paired remarks: verbal correct & numeric wrong %d, numeric correct & verbal wrong %d, both correct %d, both wrong %d"
          % (sum(v["verbal"] and not v["numeric"] for v in pairs), sum(v["numeric"] and not v["verbal"] for v in pairs),
             sum(v["verbal"] and v["numeric"] for v in pairs), sum((not v["verbal"]) and (not v["numeric"]) for v in pairs)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary()
    else:
        run()
        summary()
