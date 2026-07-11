# IRT-Viz-Eval Anonymous Review Supplement

This archive accompanies the anonymized manuscript, "How Large Language Model
Benchmarks Are Built: A Measurement-Oriented Tutorial with IRT-Viz-Eval."

It contains the benchmark generator, generated stimuli and task records, JSON
schemas, prompts, deterministic baselines and judgments, analysis outputs,
tests, and figure-generation code. It contains no API key, paid API output,
author metadata, Git history, submission correspondence, or downloaded source
articles.

Run the offline benchmark pipeline with:

```bash
python3 -m pip install -r requirements.txt
bash scripts/run_local_benchmark.sh
```

Run the tests with:

```bash
python3 -m pip install -e '.[test]'
pytest -q
```

The diagnostic `oracle`, `noisy`, and `blind` responses validate software
mechanics. They are not outputs from actual language models. The optional API
adapter requires a locally supplied key and model identifiers; no credentials
or paid-run outputs are included in this review archive.
