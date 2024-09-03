"""
pip install .[dev]
pytest .
See https://removenikud.dicta.org.il/
See https://www.nakdimon.org/
"""
import pytest
from pathlib import Path
from nakdimon_ort import Nakdimon
import difflib


# Define a fixture to initialize Nakdimon
@pytest.fixture(scope="module")
def nakdimon():
    config_path = Path(__file__).parent / '../assets/config.json'
    return Nakdimon('nakdimon.onnx', config_path)

# Function to get all test files
def get_test_files():
    files_path = Path(__file__).parent / 'files'
    return [file for file in files_path.glob('*')]

@pytest.mark.parametrize('test_file', get_test_files(), ids=lambda x: x.name)
def test_diacritics(nakdimon: 'Nakdimon', test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        text = f.read()    
    dotted_text = nakdimon.compute(text)
    
    if dotted_text != text:
        diff = difflib.unified_diff(
            text.splitlines(),
            dotted_text.splitlines(),
            fromfile='Expected',
            tofile='Got',
            lineterm=''
        )
        diff_output = '\n'.join(diff)
        
        raise AssertionError(
            f"Failed for file {test_file.name}.\n"
            f"Expected length: {len(text)} characters, Got length: {len(dotted_text)} characters\n"
            f"Diff:\n{diff_output}"
        )