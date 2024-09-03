# Building

## Build

```console
python -m pip install -U build
python -m build
```

## Publish

*Get token from https://pypi.org/manage/account/token/ *

```console
python -m pip install -U twine

python -m build
python -m twine upload dist/*
```

_Add `--repository testpypi` to upload to test repository_

## Test

```console
pip install .[dev]
pytest -s .
```