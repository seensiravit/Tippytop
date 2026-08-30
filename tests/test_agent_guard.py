from tippytop.agent.guard import scan
from tippytop.agent.contract import seed_solution_source


def test_clean_seed_passes():
    assert scan(seed_solution_source()) == []


def test_flags_test_split_access():
    assert scan("x = data.splits['test']")
    assert scan('y = data.X("test")')
    assert scan("model.predict(data, split='test')")


def test_flags_network_imports():
    assert scan("import socket")
    assert scan("import requests")
    assert scan("from urllib.request import urlopen")


def test_flags_dangerous_calls():
    assert scan("eval('1+1')")
    assert scan("import os; os.system('dir')")


def test_syntax_error_reported():
    v = scan("def broken(:\n  pass")
    assert any("syntax error" in x for x in v)
