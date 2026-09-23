"""Divergence analysis across repeated HTR runs.

Given N transcripts of the same page, pick a reference (the medoid — the run
most similar to all others), align every other run to it character-by-character,
and derive agreement scores at the character and word level. Low agreement marks
the places where the model is least stable, i.e. the likeliest reading errors.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher


def _matcher(a: str, b: str) -> SequenceMatcher:
    # autojunk misfires on long texts full of frequent characters (spaces),
    # silently degrading the alignment
    return SequenceMatcher(None, a, b, autojunk=False)


def choose_reference(transcripts: list[str]) -> int:
    """Index of the medoid transcript (highest total similarity to the rest)."""
    if len(transcripts) == 1:
        return 0
    best_i, best_score = 0, float("-inf")
    for i, t in enumerate(transcripts):
        score = sum(
            _matcher(t, u).ratio() for j, u in enumerate(transcripts) if j != i
        )
        if score > best_score:
            best_i, best_score = i, score
    return best_i


def _ref_to_run_map(ref: str, run: str) -> tuple[list[int], list[bool], list[int]]:
    """Align `run` against `ref`.

    Returns:
      mapping: for each ref char index (plus one sentinel at the end), the
               corresponding char index in `run` — used to pull out what this
               run wrote in place of any ref span.
      agree:   per ref char, whether this run has the identical char here.
      inserts: per ref gap position (len(ref)+1), chars this run inserted there.
    """
    n = len(ref)
    mapping = [0] * (n + 1)
    agree = [False] * n
    inserts = [0] * (n + 1)

    for tag, i1, i2, j1, j2 in _matcher(ref, run).get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                mapping[k] = j1 + (k - i1)
                agree[k] = True
        elif tag == "replace":
            span_ref, span_run = i2 - i1, j2 - j1
            for k in range(i1, i2):
                # proportional mapping inside a replaced block
                mapping[k] = j1 + min(span_run, (k - i1) * span_run // max(1, span_ref))
        elif tag == "delete":
            for k in range(i1, i2):
                mapping[k] = j1
        elif tag == "insert":
            inserts[i1] += j2 - j1
    mapping[n] = len(run)
    return mapping, agree, inserts


def analyze(transcripts: list[str]) -> dict:
    """Full divergence analysis. Returns a JSON-serializable dict."""
    transcripts = [unicodedata.normalize("NFC", t.strip()) for t in transcripts]
    n_runs = len(transcripts)
    ref_idx = choose_reference(transcripts)
    ref = transcripts[ref_idx]

    others = [t for i, t in enumerate(transcripts) if i != ref_idx]
    alignments = [_ref_to_run_map(ref, run) for run in others]

    # per-char agreement count (the reference always agrees with itself)
    char_counts = [1] * len(ref)
    insert_counts = [0] * (len(ref) + 1)
    for _, agree, inserts in alignments:
        for k, ok in enumerate(agree):
            if ok:
                char_counts[k] += 1
        for k, ins in enumerate(inserts):
            if ins:
                insert_counts[k] += 1

    # pairwise similarity matrix (for the stats view)
    pairwise = [[1.0] * n_runs for _ in range(n_runs)]
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            r = round(_matcher(transcripts[i], transcripts[j]).ratio(), 4)
            pairwise[i][j] = pairwise[j][i] = r

    # walk the reference line by line, word by word
    lines = []
    total_words = 0
    full_agreement = 0
    conf_sum = 0.0
    flagged = 0

    offset = 0
    for line_text in ref.split("\n"):
        line_words = []
        i = 0
        while i < len(line_text):
            if line_text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(line_text) and not line_text[j].isspace():
                j += 1
            ws, we = offset + i, offset + j
            word = line_text[i:j]

            # what each run wrote in place of this word
            variant_counts: dict[str, int] = {word: 1}
            variants_by_run = []
            for run_text, (mapping, _, _) in zip(others, alignments):
                sub = run_text[mapping[ws]:mapping[we]].strip()
                variants_by_run.append(sub)
                variant_counts[sub] = variant_counts.get(sub, 0) + 1

            match_count = 1 + sum(1 for v in variants_by_run if v == word)
            conf = match_count / n_runs
            chars = [round(char_counts[k] / n_runs, 3) for k in range(ws, we)]
            # an insertion inside or at the edges of the word is also divergence
            ins_here = max(insert_counts[k] for k in range(ws, we + 1))

            variants = sorted(
                ({"text": t, "count": c} for t, c in variant_counts.items()),
                key=lambda v: -v["count"],
            )
            line_words.append(
                {
                    "text": word,
                    "conf": round(conf, 3),
                    "chars": chars,
                    "inserts": ins_here,
                    "variants": variants,
                }
            )
            total_words += 1
            conf_sum += conf
            if conf == 1.0 and all(c == 1.0 for c in chars):
                full_agreement += 1
            if conf < 0.5:
                flagged += 1
            i = j
        lines.append({"words": line_words})
        offset += len(line_text) + 1  # the split-off "\n"

    return {
        "reference_index": ref_idx,
        "num_runs": n_runs,
        "transcripts": transcripts,
        "lines": lines,
        "pairwise": pairwise,
        "stats": {
            "total_words": total_words,
            "mean_word_conf": round(conf_sum / total_words, 4) if total_words else 0,
            "full_agreement": full_agreement,
            "flagged": flagged,
            "line_count": len(lines),
            "run_line_counts": [t.count("\n") + 1 if t else 0 for t in transcripts],
        },
    }
