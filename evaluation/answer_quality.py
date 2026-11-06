"""End-to-end answer-quality evaluation.

For every question the real pipeline is run (retrieve, filter, generate) and
three things are checked:

  context hit   the evidence needed to answer was in the context the model
                received (answerable questions)
  answer ok     the generated answer satisfies the regex checks
                (answerable questions)
  refused       the question was rejected, or the model declined to answer
                (unanswerable questions)

These are simple automatic checks, not human judgement: an answer can pass
the regexes and still be poorly worded, or fail them while being acceptable.
Use the numbers to compare versions of the pipeline, not as an absolute score.

    python -m evaluation.answer_quality                 # dev set
    python -m evaluation.answer_quality --set heldout   # or: final
    python -m evaluation.answer_quality --set all --details
"""

import argparse
import re
import time

from evaluation.answer_quality_dataset import DEV_SET, FINAL_SET, HELDOUT_SET


REFUSAL = re.compile(
    r"not enough information|enough information|insufficient|"
    r"(do|does|did) not (contain|provide|have|mention|include|specify)|"
    r"(don't|doesn't) (have|contain|provide|mention|include|specify)|"
    r"cannot (find|answer)|could not find|unable to|"
    r"no (relevant |specific )?information|"
    r"not (mentioned|provided|covered|specified|available)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def score(item: dict, result: dict) -> dict:
    """Score one pipeline result against one dataset item."""

    # Everything the model was given: the search results and the passages
    # that continue them.
    passages = result["sources"] + result.get("adjacent", [])
    context = normalize(" ".join(p["text"] for p in passages))
    answer = result["answer"]

    row = {
        "question": item["question"],
        "answerable": item["answerable"],
        "sources": len(result["sources"]),
        "answer": answer,
    }

    if item["answerable"]:
        row["context_hit"] = all(e in context for e in item["evidence"])
        row["answer_ok"] = all(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in item["answer"]
        )
    else:
        row["refused"] = not result["sources"] or bool(REFUSAL.search(answer))
        row["rejected_before_llm"] = not result["sources"]

    return row


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    def rate(selected, key):
        return sum(r[key] for r in selected) / len(selected) if selected else 0

    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "context_hit": rate(answerable, "context_hit"),
        "answer_ok": rate(answerable, "answer_ok"),
        "refused": rate(unanswerable, "refused"),
        "rejected_before_llm": rate(unanswerable, "rejected_before_llm"),
        "avg_sources": (
            sum(r["sources"] for r in answerable) / len(answerable)
            if answerable else 0
        ),
    }


def run(items: list[dict], ask=None) -> list[dict]:
    """Run the pipeline on every item. `ask` defaults to the real pipeline."""

    if ask is None:
        from backend.rag_pipeline import ask_question as ask

    rows = []

    for item in items:
        started = time.time()
        row = score(item, ask(item["question"]))
        row["seconds"] = time.time() - started
        rows.append(row)

    return rows


def print_summary(title: str, rows: list[dict]):
    s = summarize(rows)

    print(f"\n{title}  ({s['answerable']} answerable, {s['unanswerable']} unanswerable)")
    print(f"  context hit      {s['context_hit']:7.2%}   (needed evidence was given to the model)")
    print(f"  answer ok        {s['answer_ok']:7.2%}   (answer passes the regex checks)")
    print(f"  refused          {s['refused']:7.2%}   (unanswerable: rejected or declined)")
    print(f"  rejected early   {s['rejected_before_llm']:7.2%}   (unanswerable: filtered before the LLM)")
    print(f"  avg sources/answer {s['avg_sources']:5.2f}")


def print_details(rows: list[dict]):
    for row in rows:
        if row["answerable"]:
            flags = f"hit={row['context_hit']!s:5} ok={row['answer_ok']!s:5}"
        else:
            flags = f"refused={row['refused']!s:5} early={row['rejected_before_llm']!s:5}"

        print(f"\n- {row['question']}\n  {flags}  sources={row['sources']}  {row['seconds']:.1f}s")
        print("  " + row["answer"][:240].replace("\n", " "))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=["dev", "heldout", "final", "all"], default="dev")
    parser.add_argument("--details", action="store_true", help="print every answer")
    args = parser.parse_args()

    items = {
        "dev": DEV_SET,
        "heldout": HELDOUT_SET,
        "final": FINAL_SET,
        "all": DEV_SET + HELDOUT_SET + FINAL_SET,
    }[args.set]

    rows = run(items)

    if args.details:
        print_details(rows)

    print_summary(f"ANSWER QUALITY - {args.set} set", rows)


if __name__ == "__main__":
    main()
