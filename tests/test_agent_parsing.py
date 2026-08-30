from tippytop.agent.parsing import parse_response

FENCE = "```"


def test_extracts_hypothesis_and_code():
    txt = f"Hypothesis: use BPR loss\n\n{FENCE}python\nprint('hi')\n{FENCE}"
    p = parse_response(txt)
    assert p.parse_ok
    assert p.hypothesis == "use BPR loss"
    assert p.code == "print('hi')"


def test_last_fence_wins():
    txt = (f"try this\n{FENCE}python\nold = 1\n{FENCE}\n"
           f"actually the full script:\n{FENCE}python\nnew = 2\n{FENCE}")
    p = parse_response(txt)
    assert p.parse_ok
    assert p.code == "new = 2"


def test_no_fence_is_not_ok():
    p = parse_response("I could not produce code this time.")
    assert not p.parse_ok
    assert p.parse_error
    assert p.code == ""


def test_bare_fence_language_tolerated():
    txt = f"h\n{FENCE}\nx = 1\n{FENCE}"
    p = parse_response(txt)
    assert p.parse_ok and p.code == "x = 1"
