# Contributing

Use the setup instructions in the [README](README.md), then run:

```sh
python tests/test_smoke.py
```

For a bug report, include the command, Python and dependency versions, traceback
and steps to reproduce it. State whether the issue needs a camera or occurs with
synthetic data.

Keep changes focused. Explain any change to feature normalization, model inputs,
checkpoint format or intent confirmation, and include a regression check when
behavior changes. Do not commit captured face data, checkpoints or generated
screenshots.

Reported experiments should identify the dataset, split, model configuration and
hardware. Distinguish synthetic checks from real-data evaluation, and document
data permissions when sharing a dataset.
