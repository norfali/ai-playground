"""Speech transcription evaluation using JiWER."""

from dataclasses import dataclass
from difflib import SequenceMatcher
from html import escape
from itertools import groupby

import jiwer


@dataclass(frozen=True)
class Evaluation:
    words: jiwer.WordOutput
    characters: jiwer.CharacterOutput


def normalize(text: str) -> str:
    """Normalise case and whitespace without changing punctuation."""
    return " ".join(text.casefold().split())


def evaluate(reference: str, hypothesis: str) -> Evaluation:
    """Compare recognised speech against a reference transcript."""
    reference = normalize(reference)
    hypothesis = normalize(hypothesis)
    return Evaluation(
        words=jiwer.process_words(reference, hypothesis),
        characters=jiwer.process_characters(reference, hypothesis),
    )


def _display_alignment(words: jiwer.WordOutput) -> list[jiwer.AlignmentChunk]:
    """Prefer similar word pairs among minimum edits with JiWER's exact counts.

    Each prefix keeps only minimum-cost paths, indexed by substitutions/deletions.
    Insertions follow from the prefix lengths. Similarity is only a tie-breaker;
    counts are bounded by JiWER, so the final path has its exact error breakdown.
    All steps move forwards, preserving token order. Equal similarity keeps the
    first path for deterministic repeated-word alignment.
    """
    reference, hypothesis = words.references[0], words.hypotheses[0]
    # Cell values map (substitutions, deletions) to (similarity, last operation).
    cells: dict[tuple[int, int], dict[tuple[int, int], tuple[float, str]]] = {
        (0, 0): {(0, 0): (0.0, "equal")}
    }
    for i in range(len(reference) + 1):
        for j in range(max(0, i - words.deletions), min(len(hypothesis), i + words.insertions) + 1):
            paths = cells.get((i, j), {})
            if not paths:
                continue
            minimum = min(s + 2 * d + j - i for s, d in paths)
            paths = {
                key: value for key, value in paths.items() if key[0] + 2 * key[1] + j - i == minimum
            }
            cells[i, j] = paths
            steps = []
            if i < len(reference) and j < len(hypothesis):
                same = reference[i] == hypothesis[j]
                similarity = (
                    0.0
                    if same
                    else SequenceMatcher(None, reference[i], hypothesis[j], autojunk=False).ratio()
                )
                steps.append(
                    (i + 1, j + 1, int(not same), 0, "equal" if same else "substitute", similarity)
                )
            if i < len(reference):
                steps.append((i + 1, j, 0, 1, "delete", 0.0))
            if j < len(hypothesis):
                steps.append((i, j + 1, 0, 0, "insert", 0.0))
            for ni, nj, ds, dd, kind, similarity in steps:
                for (s, d), (score, _) in paths.items():
                    ns, nd = s + ds, d + dd
                    insertions = nj - ni + nd
                    hits = ni - ns - nd
                    if not (
                        ns <= words.substitutions
                        and nd <= words.deletions
                        and insertions <= words.insertions
                        and hits <= words.hits
                    ):
                        continue
                    destination = cells.setdefault((ni, nj), {})
                    previous = destination.get((ns, nd))
                    if previous is None or score + similarity > previous[0]:
                        destination[ns, nd] = (score + similarity, kind)

    i, j = len(reference), len(hypothesis)
    s, d = words.substitutions, words.deletions
    chunks = []
    while i or j:
        _, kind = cells[i, j][s, d]
        ni = i - (kind != "insert")
        nj = j - (kind != "delete")
        chunks.append(jiwer.AlignmentChunk(kind, ni, i, nj, j))
        s -= kind == "substitute"
        d -= kind == "delete"
        i, j = ni, nj
    return _group_gaps(list(reversed(chunks)))


def _group_gaps(ordered: list[jiwer.AlignmentChunk]) -> list[jiwer.AlignmentChunk]:
    """Keep consecutive extra/missing words together, and word pairs separate."""
    grouped = []
    for kind, run in groupby(ordered, key=lambda chunk: chunk.type):
        span = list(run)
        if kind in {"insert", "delete"}:
            first, last = span[0], span[-1]
            grouped.append(
                jiwer.AlignmentChunk(
                    kind,
                    first.ref_start_idx,
                    last.ref_end_idx,
                    first.hyp_start_idx,
                    last.hyp_end_idx,
                )
            )
        else:
            grouped.extend(span)
    return grouped


def _highlight_characters(text: str, changed: set[int]) -> str:
    """Highlight changed positions in escaped, normalised text."""
    parts = []
    for highlighted, group in groupby(enumerate(text), key=lambda item: item[0] in changed):
        content = escape("".join(character for _, character in group))
        parts.append(f'<strong class="diff-chars">{content}</strong>' if highlighted else content)
    return "".join(parts)


def _render_substitution(reference: str, recognized: str) -> tuple[str, str]:
    """Highlight differences between normalised tokens; leave JiWER scoring untouched."""
    reference_changed: set[int] = set()
    recognized_changed: set[int] = set()
    for kind, start, end, other_start, other_end in SequenceMatcher(
        None, reference, recognized, autojunk=False
    ).get_opcodes():
        if kind != "equal":
            reference_changed.update(range(start, end))
            recognized_changed.update(range(other_start, other_end))
    return (
        _highlight_characters(reference, reference_changed),
        _highlight_characters(recognized, recognized_changed),
    )


def _render_difference_row(kind: str, reference: str, recognized: str) -> str:
    """Render an escaped error row, with character emphasis for substitutions."""
    labels = {
        "substitute": "Different word",
        "delete": "Missing word",
        "insert": "Extra word",
    }
    if kind == "substitute":
        reference_html, recognized_html = _render_substitution(reference, recognized)
    else:
        reference_html = escape(reference) or '<span class="diff-absent">No reference words</span>'
        recognized_html = escape(recognized) or '<span class="diff-absent">Missing words</span>'
    return (
        f'<tr class="diff-{kind}"><th scope="row">{labels[kind]}</th>'
        f"<td>{reference_html}</td><td>{recognized_html}</td></tr>"
    )


def render_differences(result: Evaluation) -> str:
    """Render display alignment independently of JiWER's scoring alignment."""
    words = result.words

    if words.wer == 0:
        return "<p>No differences found</p>"

    rows = []
    for chunk in _display_alignment(words):
        if chunk.type == "equal":
            continue
        reference = words.references[0][chunk.ref_start_idx : chunk.ref_end_idx]
        recognized = words.hypotheses[0][chunk.hyp_start_idx : chunk.hyp_end_idx]
        rows.append(_render_difference_row(chunk.type, " ".join(reference), " ".join(recognized)))

    return (
        '<div class="comparison" role="region" aria-label="Where they differed" tabindex="0">'
        '<table class="diff-table" aria-label="Transcript differences">'
        '<thead><tr><th scope="col">Difference</th><th scope="col">Reference</th>'
        '<th scope="col">Whisper output</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def describe_metrics(result: Evaluation) -> str:
    """Render semantic HTML metrics; rates require a nonempty normalised reference."""
    words = result.words
    has_word_reference = bool(words.references[0])
    has_character_reference = bool(result.characters.references[0])
    wer = f"{words.wer:.1%}" if has_word_reference else "—"
    cer = f"{result.characters.cer:.1%}" if has_character_reference else "—"
    errors = words.substitutions + words.deletions + words.insertions
    metrics = (
        '<dl class="metric-list" aria-label="Comparison metrics">'
        f"<div><dt>WER</dt><dd>{wer}</dd></div>"
        f"<div><dt>CER</dt><dd>{cer}</dd></div>"
        f"<div><dt>Word errors</dt><dd>{errors}</dd></div></dl>"
        f'<p class="error-counts">Different words: {words.substitutions} · '
        f"Missing words: {words.deletions} · Extra words: {words.insertions}</p>"
    )
    if not has_word_reference and not has_character_reference:
        metrics += (
            f"<p>Reference is empty. Whisper inserted {words.insertions} words. "
            "Rates need a reference.</p>"
        )
    return metrics
