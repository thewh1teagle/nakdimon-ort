"""
pip install .[dev]
pytest .
See https://removenikud.dicta.org.il/
"""
import pytest
from pathlib import Path
from nakdimon_ort import Nakdimon

# Define a fixture to initialize Nakdimon
@pytest.fixture(scope="module")
def nakdimon():
    return Nakdimon('nakdimon.onnx', 'config.json')

# Function to get all test files
def get_test_files():
    files_path = Path(__file__).parent / 'files'
    return [file for file in files_path.glob('*')]

@pytest.mark.parametrize('test_file', get_test_files(), ids=lambda x: x.name)
def test_diacritics(nakdimon: 'Nakdimon', test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        text = f.read()    
    dotted_text = nakdimon.compute(text)
    
    assert dotted_text == text, f"Failed for file {test_file.name}. Expected: {text} but got: {dotted_text}"