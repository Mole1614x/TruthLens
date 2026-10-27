import pandas as pd

from text_cleaner import clean_text

def test_lowercase():
    assert clean_text("Breaking News") == "breaking news"

def test_whitespace():
    assert clean_text("  fake   news\n today  ") == "fake news today"

def test_missing_value():
    assert clean_text(pd.NA) == ""

def test_punctuation_preserved():
    assert clean_text("This isn't true!") == "this isn't true!"

if __name__ == "__main__":
    test_lowercase()
    test_whitespace()
    test_missing_value()
    test_punctuation_preserved()

    print("All text cleaner tests passed.")