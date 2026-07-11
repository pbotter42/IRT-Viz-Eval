# Contributing

Contributions that improve psychometric coverage, provider adapters, scoring
validation, accessibility, or documentation are welcome.

1. Create a focused branch.
2. Install development dependencies with `python3 -m pip install -e '.[test]'`.
3. Add or update tests for behavioral changes.
4. Run `pytest` and the local benchmark pipeline.
5. Open a pull request describing the intended inference affected by the change.

Do not submit API keys, private prompts, copyrighted source articles, or paid-run
outputs that have not been reviewed for release. New benchmark tasks should
include independently generated or verified ground truth and a clear scoring
rule.
