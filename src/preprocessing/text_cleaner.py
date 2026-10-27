
import re
import pandas as pd

def clean_text(text):
    """
    Clean a text value for the initial TF-IDF baseline.
    """

    # Handle missing values
    if pd.isna(text):
        return ""

    # Ensure input is a string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text
