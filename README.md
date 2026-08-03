# 3REx
Tree Rules Extractor (3REx) automatically extracts rules from pre-trained *sklearn.ensemble* **RandomForestClassifier** models.

## Modes
- `--mode validate`: validates and checks inputs
- `--mode inspect`: reports information about the trees
- `--mode extract`: extracts decisional tree rules and most frequent rules
- `--mode frequent-rules`: extracts most frequent rules from a previously generated *rules.json* file. The latter must be provided through the --rules path-to-rules.json argument.

## Input files
Suppose having the following input files in the *'input/'* directory:
- model.pkl
- feature_names.csv
- scalers.pkl
- rules.json

Not all files are required for every mode.

**NOTE: if the `--features input/feature_names.csv` file is not provided, 3REx attempts to recover the feature names directly from the --model model.pkl file.**

## Usage
### Inspecting trees
`python treerex.py --mode inspect --model input/model.pkl --features input/feature_names.csv --max-trees 10 --visualization-depth 1000`

This command inspects the first 10 trees in the forest while producing a graphical representation in both `.dot` and `.png` formats.
The maximum visualization depth is set to 1000.

### Extracting rules

`python treerex.py --mode extract --model input/model.pkl`

This command extracts all the decision rules produced by the Random Forest and identifies the most frequent rules.

### Extracting rules with scalers

`python treerex.py --mode extract --model input/model.pkl --scalers input/scalers.pkl`

The command extracts all the rules produced by the model.
When provided with the pre-processing scalers object (`--scalers input/scalers.pkl`) 3REx also recovers the thresholds in the original feature scale.
Therefore, both the model-scale rules and a human-readable version, containing the recovered original thresholds, will be produced.

### Analysing previously extracted rules
`python treerex.py --mode frequent-rules --rules input/rules.json`

This command identifies the most frequent exact rules from an existing `--rules input/rules.json` file.

By default, the order of conditions connected by `AND` is ignored when comparing rules. 

Therefore 
**A AND B** == **B AND A** meaning that they are treated as **equivalent rules**.

This behaviour can be disabled in the source code by setting `ignore_condition_order=False`.
When condition order is preserved, **A AND B** and **B AND A** are treated as **different rules**.

### Custom output directory

A custom output directory can be specified in any mode through the `--output-dir` argument. E.g.:

`python treerex.py --mode extract --model input/model.pkl --output-dir custom_output`

