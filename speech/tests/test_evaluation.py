"""Tests for transcription accuracy and comparison rendering."""

import re
from dataclasses import replace
from difflib import SequenceMatcher
from html import escape
from itertools import product

import pytest

from speech_lab.evaluation import (
    _display_alignment,
    describe_metrics,
    evaluate,
    normalize,
    render_differences,
)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "wer", "cer"),
    [
        ("hello world", "hello world", 0, 0),
        ("  Hello\n WORLD ", "hello world", 0, 0),
        ("", "", 0, 0),
        (" \n\t", "", 0, 0),
        ("", "hi all", 2, 6),
        ("hello", "", 1, 1),
        ("hello!", "hello", 1, 1 / 6),
        ("a", "a b c", 2, 4),
    ],
)
def test_metrics(reference, hypothesis, wer, cer):
    result = evaluate(reference, hypothesis)

    assert result.words.wer == pytest.approx(wer)
    assert result.characters.cer == pytest.approx(cer)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "kind"),
    [
        ("the cat", "the dog", "substitute"),
        ("the big cat", "the cat", "delete"),
        ("the cat", "the big cat", "insert"),
    ],
)
def test_alignment(reference, hypothesis, kind):
    result = evaluate(reference, hypothesis)

    errors = [chunk for chunk in result.words.alignments[0] if chunk.type != "equal"]

    assert len(errors) == 1
    assert errors[0].type == kind
    assert (result.words.substitutions + result.words.deletions + result.words.insertions) == 1


def test_normalization():
    assert normalize("  CAFÉ,\ncan't!  ") == "café, can't!"


def test_rendering_escapes_html():
    html = render_differences(evaluate("safe", "<script>alert('x')</script>"))

    assert "<script>" not in html
    assert "&lt;script&gt;" in re.sub(r"</?strong\b[^>]*>", "", html)
    assert "Different word" in html


def test_identical_transcripts():
    html = render_differences(evaluate("hello", "hello"))

    assert html == "<p>No differences found</p>"


@pytest.mark.parametrize("reference", ["", " \n\t\u00a0\u2003"])
def test_empty_reference(reference):
    result = evaluate(reference, "hi all")
    description = describe_metrics(result)

    assert "2 words" in description
    assert "Reference is empty" in description
    assert "<dt>WER</dt><dd>—</dd>" in description
    assert "<dt>CER</dt><dd>—</dd>" in description
    assert "<td>hi all</td>" in render_differences(result)


@pytest.mark.parametrize("word_reference", [False, True])
def test_metric_denominators_are_checked_independently(word_reference):
    empty = evaluate("", "x")
    nonempty = evaluate("a", "b")
    result = replace(
        nonempty if word_reference else empty,
        characters=(empty if word_reference else nonempty).characters,
    )

    description = describe_metrics(result)

    assert f"<dt>WER</dt><dd>{'100.0%' if word_reference else '—'}</dd>" in description
    assert f"<dt>CER</dt><dd>{'—' if word_reference else '100.0%'}</dd>" in description


def test_error_breakdown_follows_primary_metrics():
    description = describe_metrics(evaluate("one missing word", "one extra"))

    assert '<dt>Word errors</dt><dd>2</dd></div></dl><p class="error-counts">' in description
    assert "Different words: 1 · Missing words: 1 · Extra words: 0" in description


@pytest.mark.parametrize(
    ("reference_error", "recognised_error", "label"),
    [
        ("CAFÉ,", "Café!", "Different word"),
        ("Été—東京", "", "Missing word"),
        ("", "İSTANBUL’s🙂", "Extra word"),
        ("<ÉTÉ>&", '"東京"', "Different word"),
    ],
)
def test_only_errors_render_with_normalized_tokens(reference_error, recognised_error, label):
    # Casefold expands ß and İ; render the same tokens JiWER aligned and scored.
    prefix = "Straße İSTANBUL a long matching passage"
    suffix = "and another matching passage at the end"
    reference = f"{prefix} {reference_error} {suffix}"
    hypothesis = f"{prefix.casefold()} {recognised_error} {suffix}"
    result = evaluate(reference, hypothesis)
    html = render_differences(result)
    plain_html = re.sub(r"</?strong\b[^>]*>", "", html)

    assert html.count('<tr class="diff-') == 1
    assert f'<th scope="row">{label}</th>' in html
    reference_html = (
        escape(normalize(reference_error)) or '<span class="diff-absent">No reference words</span>'
    )
    recognised_html = (
        escape(normalize(recognised_error)) or '<span class="diff-absent">Missing words</span>'
    )
    assert f"<td>{reference_html}</td><td>{recognised_html}</td>" in plain_html
    if label != "Different word":
        assert '<strong class="diff-chars">' not in html
    assert 'class="diff-equal"' not in html
    assert "matching passage" not in html
    assert result.words.wer == pytest.approx(1 / len(reference.split()))


def test_case_and_whitespace_only_changes_have_no_differences():
    result = evaluate("  Straße\nİSTANBUL CAFÉ, ", "STRASSE  i̇stanbul café,")

    assert result.words.wer == result.characters.cer == 0
    assert render_differences(result) == "<p>No differences found</p>"


def test_unicode_and_whitespace_errors_render_jiwer_tokens():
    result = evaluate(" \tKEEP\u00a0Straße\nİX\u2003END ", "keep\nSTRASZE\tİY end")

    assert result.words.references[0] == ["keep", "strasse", "i̇x", "end"]
    assert result.words.hypotheses[0] == ["keep", "strasze", "i̇y", "end"]
    html = render_differences(result)
    plain_html = re.sub(r"</?strong\b[^>]*>", "", html)

    assert html.count('<tr class="diff-substitute">') == 2
    assert "<td>strasse</td><td>strasze</td>" in plain_html
    assert "<td>i̇x</td><td>i̇y</td>" in plain_html
    assert "keep" not in html
    assert "end" not in html


def test_multiple_error_tokens_use_normalized_spelling():
    html = render_differences(evaluate("HELLO, 東京!", "Goodbye… ΚΌΣΜΕ?"))
    plain_html = re.sub(r"</?strong\b[^>]*>", "", html)

    assert "<td>hello,</td><td>goodbye…</td>" in plain_html
    assert "<td>東京!</td><td>κόσμε?</td>" in plain_html
    assert html.count('<tr class="diff-substitute">') == 2


@pytest.mark.parametrize(
    ("reference", "hypothesis", "reference_html", "hypothesis_html"),
    [
        (
            "floor-fattened",
            "flour-fattened",
            'flo<strong class="diff-chars">o</strong>r-fattened',
            'flo<strong class="diff-chars">u</strong>r-fattened',
        ),
        ("Hello!", "hello", 'hello<strong class="diff-chars">!</strong>', "hello"),
        ("hello", "hello?!", "hello", 'hello<strong class="diff-chars">?!</strong>'),
        (
            "CAFÉ,",
            "cafe!",
            'caf<strong class="diff-chars">é,</strong>',
            'caf<strong class="diff-chars">e!</strong>',
        ),
        (
            "Straße🙂",
            "STRASSE🙃",
            'strasse<strong class="diff-chars">🙂</strong>',
            'strasse<strong class="diff-chars">🙃</strong>',
        ),
        (
            "İ東京",
            "i̇東亰",
            'i̇東<strong class="diff-chars">京</strong>',
            'i̇東<strong class="diff-chars">亰</strong>',
        ),
        ("café", "cafe", 'cafe<strong class="diff-chars">́</strong>', "cafe"),
        (
            "a<b&\"'",
            "a>b&\"'",
            'a<strong class="diff-chars">&lt;</strong>b&amp;&quot;&#x27;',
            'a<strong class="diff-chars">&gt;</strong>b&amp;&quot;&#x27;',
        ),
    ],
)
def test_substitution_character_highlighting(
    reference, hypothesis, reference_html, hypothesis_html
):
    html = render_differences(evaluate(reference, hypothesis))

    assert f"<td>{reference_html}</td><td>{hypothesis_html}</td>" in html
    assert 'aria-label="Where they differed"' in html
    assert '<table class="diff-table" aria-label="Transcript differences">' in html


def test_adjacent_substitutions_highlight_each_word():
    result = evaluate("cat floor", "cats flour")
    assert len(result.words.alignments[0]) == 1
    assert result.words.substitutions == 2
    metrics = describe_metrics(result)
    html = render_differences(result)

    assert html.count('<tr class="diff-substitute">') == 2
    assert '<td>cat</td><td>cat<strong class="diff-chars">s</strong></td>' in html
    assert (
        '<td>flo<strong class="diff-chars">o</strong>r</td>'
        '<td>flo<strong class="diff-chars">u</strong>r</td>'
    ) in html
    assert describe_metrics(result) == metrics
    assert result.words.wer == 1
    assert result.characters.cer == pytest.approx(2 / 9)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "label", "absent"),
    [
        ("<script>& café", "", "Missing word", "Missing words"),
        ("", "<script>& café", "Extra word", "No reference words"),
    ],
)
def test_missing_and_additional_words_are_labelled_and_escaped(
    reference, hypothesis, label, absent
):
    html = render_differences(evaluate(reference, hypothesis))

    assert label in html
    assert f'<span class="diff-absent">{absent}</span>' in html
    assert "&lt;script&gt;&amp; café" in html
    assert "<script>" not in html


def test_many_errors_share_one_table_in_alignment_order():
    reference = "cat keep missing stay anchor end " + " ".join(f"a{i} same{i}" for i in range(20))
    hypothesis = "dog keep stay anchor extra end " + " ".join(f"b{i} same{i}" for i in range(20))
    result = evaluate(reference, hypothesis)
    html = render_differences(result)
    expected_types = ["substitute", "delete", "insert"] + ["substitute"] * 20

    assert html.count("<table ") == 1
    assert html.count('<tr class="diff-') == len(expected_types)
    assert re.findall(r'<tr class="diff-([^"]+)">', html) == expected_types
    assert "keep" not in html
    assert "same19" not in html
    assert "<ul" not in html


def test_table_has_accessible_headers_and_keyboard_scroll_region():
    html = render_differences(evaluate("cat", "dog"))

    assert 'role="region" aria-label="Where they differed" tabindex="0"' in html
    assert (
        '<thead><tr><th scope="col">Difference</th><th scope="col">Reference</th>'
        '<th scope="col">Whisper output</th></tr></thead>'
    ) in html
    assert '<th scope="row">Different word</th>' in html
    assert 'aria-label="Transcript differences"' in html
    assert "<caption>" not in html
    assert "Underlined" not in html


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        (
            "peppered, floor-fattened.",
            "peppered, flour-fattened sauce.",
            [("substitute", "floor-fattened.", "flour-fattened"), ("insert", "", "sauce.")],
        ),
        (
            "peppered, flour-fattened sauce.",
            "peppered, floor-fattened.",
            [("substitute", "flour-fattened", "floor-fattened."), ("delete", "sauce.", "")],
        ),
        (
            "cat floor cat",
            "cat flour sauce cat",
            [("substitute", "floor", "flour"), ("insert", "", "sauce")],
        ),
        ("cat cat", "cat", [("delete", "cat", "")]),
        ("cat", "cat cat", [("insert", "", "cat")]),
        (
            "<cat> floor 東京",
            "<cats> flour",
            [
                ("substitute", "<cat>", "<cats>"),
                ("substitute", "floor", "flour"),
                ("delete", "東京", ""),
            ],
        ),
        (
            "cat floor",
            "cats flour extra words",
            [
                ("substitute", "cat", "cats"),
                ("substitute", "floor", "flour"),
                ("insert", "", "extra words"),
            ],
        ),
        # Pairing cat/cats would increase the minimum edit cost from one to two.
        ("cat dog", "cats cat dog", [("insert", "", "cats")]),
        # Another cost-two alignment exists (delete/equal/insert), but its counts differ.
        ("cat dog", "dog cats", [("substitute", "cat", "dog"), ("substitute", "dog", "cats")]),
    ],
)
def test_display_alignment_regressions(reference, hypothesis, expected):
    result = evaluate(reference, hypothesis)
    original = repr(result)
    alignment = _display_alignment(result.words)
    actual = [
        (
            chunk.type,
            " ".join(result.words.references[0][chunk.ref_start_idx : chunk.ref_end_idx]),
            " ".join(result.words.hypotheses[0][chunk.hyp_start_idx : chunk.hyp_end_idx]),
        )
        for chunk in alignment
        if chunk.type != "equal"
    ]
    assert actual == expected
    assert _display_alignment(result.words) == alignment
    html = re.sub(r"</?strong\b[^>]*>", "", render_differences(result))
    for kind, ref, hyp in expected:
        assert f'<tr class="diff-{kind}">' in html
        if ref:
            assert f"<td>{escape(ref)}</td>" in html
        if hyp:
            assert f"<td>{escape(hyp)}</td>" in html
    assert repr(result) == original


def test_display_alignment_matches_exhaustive_minimum_edit_oracle():
    """Enumerate short paths independently, including repeated-word ambiguities."""

    def paths(ref, hyp):
        if not ref and not hyp:
            yield (0, 0, 0), 0.0
        if ref and hyp:
            different = ref[0] != hyp[0]
            similarity = (
                SequenceMatcher(None, ref[0], hyp[0], autojunk=False).ratio() if different else 0.0
            )
            for (s, d, i), score in paths(ref[1:], hyp[1:]):
                yield (s + different, d, i), score + similarity
        if ref:
            for (s, d, i), score in paths(ref[1:], hyp):
                yield (s, d + 1, i), score
        if hyp:
            for (s, d, i), score in paths(ref, hyp[1:]):
                yield (s, d, i + 1), score

    sequences = [
        tokens for length in range(4) for tokens in product(("cat", "cats"), repeat=length)
    ]
    for reference, hypothesis in product(sequences, repeat=2):
        words = evaluate(" ".join(reference), " ".join(hypothesis)).words
        counts = (words.substitutions, words.deletions, words.insertions)
        alternatives = list(paths(reference, hypothesis))
        assert sum(counts) == min(sum(count) for count, _ in alternatives)
        best = max(score for count, score in alternatives if count == counts)
        displayed = {"substitute": 0, "delete": 0, "insert": 0}
        score = 0.0
        ref_end = hyp_end = 0
        for chunk in _display_alignment(words):
            assert (chunk.ref_start_idx, chunk.hyp_start_idx) == (ref_end, hyp_end)
            ref_end, hyp_end = chunk.ref_end_idx, chunk.hyp_end_idx
            ref = reference[chunk.ref_start_idx : ref_end]
            hyp = hypothesis[chunk.hyp_start_idx : hyp_end]
            if chunk.type == "equal":
                assert ref == hyp
            elif chunk.type == "substitute":
                assert len(ref) == len(hyp) == 1
                displayed[chunk.type] += 1
                score += SequenceMatcher(None, ref[0], hyp[0], autojunk=False).ratio()
            else:
                assert not (ref if chunk.type == "insert" else hyp)
                displayed[chunk.type] += len(ref) + len(hyp)
        assert (ref_end, hyp_end) == (len(reference), len(hypothesis))
        assert tuple(displayed.values()) == counts
        assert score == pytest.approx(best)
