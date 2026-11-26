import pickle
import argparse
from pathlib import Path
from typing import Sequence
import csv
import logging
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import json
from datetime import datetime
import subprocess
import math
import re


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="3REx",
        description=("3REx extracts human-readable rules from pre-trained Random Forest models.")
    )

    parser.add_argument(
        "--model",
        type=Path,
        #required=True,
        default=None,
        help="Path to the Random Forest model. Required for the validate, inspect and extract modalities."
    )

    parser.add_argument(
        "--scalers",
        type=Path,
        default = None,
        help="Optional path to the .pkl file containing the scalers used during the training."
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help=("Optional path to the .csv file containing the ordered feature names. If absent, names are recovered from the model or generated automatically.")
    )

    parser.add_argument(
        "--rules",
        type=Path,
        default=None,
        help=("Optional path to a rules.json file. Required when mode is 'frequent-rules'.")
    )

    parser.add_argument(
        "--mode",
        choices=["validate", "inspect", "extract", "frequent-rules"],
        default="validate",
        help="Execution mode: 'validate' only checks inputs; 'inspect' prints basic tree information; 'extract' extracts decisional tree rules; 'frequent-rules' extracts the most frequent rules from a .json file provive through the --rules arg."
    )

    parser.add_argument(
        "--max-trees",
        type=int,
        default=1,
        help="Maximum number of trees to inspect."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where 3REx output files are saved."
    )

    parser.add_argument(
        "--visualization-depth",
        type=int,
        default=4,
        help="Maximum tree depth shown in the generated graph visualization."
    )

    arguments = parser.parse_args()
    return arguments

def validate_input(path: Path, description : str):
    """
    Verifies whether the input path exists and is referring to a real file. Otherwise, it raises and exception.
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a real file: {path}")

def validate_model_features(model: RandomForestClassifier, feature_names: Sequence[str]):
    """
    Verifies whether the total model's features number is the same as the feature_names one.
    """
    model_feature_count = int(model.n_features_in_)
    input_feature_count = len(feature_names)

    if model_feature_count != input_feature_count:
        raise ValueError(f"Inconsistency: model expects {model_feature_count} features, but the feature file contains {input_feature_count} rows.")

def _expand_condition(missing_go_to_left, threshold, feature_name, node_id, left_child, right_child):
    if math.isnan(threshold):
        split_type = "missing"
        threshold_for_csv = ""
        if missing_go_to_left:
            condition = f"{feature_name} IS MISSING"
        else:
            condition = f"{feature_name} IS NOT MISSING"
        logger.info(
            f"Node {node_id}: MISSING SPLIT - if {condition} then go to node {left_child} else go to node {right_child}")
    else:
        split_type = "numeric"
        threshold_for_csv = threshold
        condition = f"{feature_name} <= {threshold:.6f}"
        logger.info(
            f"Node {node_id}: NUMERIC SPLIT - if {condition} then go to node {left_child} else go to node {right_child}")
    return split_type, threshold_for_csv, condition

def load_scalers(scalers_path: Path):
    """
    Loads the calers file and verifies the external structure
    """
    logger.info(f"Loading scalers from {scalers_path}")

    with scalers_path.open("rb") as file:
        scalers = pickle.load(file)

    if not isinstance(scalers, (list, tuple)):
        raise TypeError(f"Expected an ordered list or tuple of preprocessing objects, found {type(scalers).__name__}.")

    return scalers

def validate_model_scalers(model, scalers):
    """
    Validates the scalers according to the model in input.
    """
    expected_count = int(model.n_features_in_)
    actual_count = len(scalers)

    if actual_count != expected_count:
        raise ValueError(f"Scaler inconsistency: model expects {expected_count} features, but the scaler file contains {actual_count} objects.")

    for feature_index, scaler in enumerate(scalers):
        if not hasattr(scaler, "transform"): # non usiamo fit_transform ma transform durante l'inferenza
            raise TypeError(f"Object at feature position {feature_index} does not provide transform().")

def validate_rules(rules):
    """
    Verifies and validates the structure required for rule analysis.
    """
    required_rule_fields = {
        "tree_index",
        "leaf_node_id",
        "conditions",
        "predicted_class_index"
    }

    required_condition_fields = {
        "feature_index",
        "operator",
        "threshold",
        #"accepts_missing"
    }

    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise TypeError(f"Rule {rule_index} must be a dictionary.")

        missing_rule_fields = required_rule_fields - rule.keys()

        if missing_rule_fields:
            raise ValueError(f"Rule {rule_index} is missing fields: {sorted(missing_rule_fields)}")

        if not isinstance(rule["conditions"], list):
            raise TypeError(f"Conditions of rule {rule_index} must be a list.")

        for condition_index, condition in enumerate(rule["conditions"]):
            if not isinstance(condition, dict):
                raise TypeError(f"Condition {condition_index} of rule {rule_index} must be a dictionary.")

            missing_condition_fields = (required_condition_fields - condition.keys())

            if missing_condition_fields:
                raise ValueError(f"Condition {condition_index} of rule {rule_index} is missing fields: {sorted(missing_condition_fields)}")

    logger.info(f"Successfully validated {len(rules)} rules.")

def _transform_condition(condition):
    """
    Transforms the representation from, e.g.,
    {feature_index: 385, operator: <=, threshold: -1.82092, accept_missing:false}
    to (384, "<=", -1.82092, False).
    This is done to normalize a condition so that can be fairly compared to others.
    """
    return (condition["feature_index"], condition["operator"], condition["threshold"], condition.get("accepts_missing"))

def _transform_rule(rule, ignore_condition_order=False):
    """
    Transforms the structure of each rule so that it can become comparable while being represented by a tuple of transformed conditions.
    When ignore_condition_order=True the following becomes true: A and B = B and A
    """
    transformed_conditions = []

    for condition in rule["conditions"]:
        transformed_condition = _transform_condition(condition)
        transformed_conditions.append(transformed_condition)

    if ignore_condition_order:
        conditions_key = frozenset(transformed_conditions)# the set becomes immutable and we can have: frozenset({A, B}) == frozenset({B, A})
    else:
        conditions_key = tuple(transformed_conditions)

    return (conditions_key, rule["predicted_class_index"])

def find_frequent_rules(rules, ignore_condition_order=False):
    """
    Groups rules having exactly the same conditions and predicted class.
    """
    total_tree_count = len({rule["tree_index"] for rule in rules})

    rule_groups = {}

    for source_rule_id, rule in enumerate(rules, start=1):
        rule_key = _transform_rule(rule, ignore_condition_order)
        if rule_key not in rule_groups:

            rule_groups[rule_key] = {
                "occurrences": 0, #first time, counter = 0 then we increment
                "tree_indices": set(),
                'matches': [],
                "representative_rule": rule
            }
        rule_groups[rule_key]["occurrences"] += 1
        rule_groups[rule_key]["tree_indices"].add(rule["tree_index"]) #each tree added only one time cause of the set
        rule_groups[rule_key]["matches"].append({
            "source_rule_id": rule.get("rule_id", source_rule_id),
            "tree_index": rule["tree_index"],
            "leaf_node_id": rule["leaf_node_id"]
        })

    frequent_rules = []
    for group in rule_groups.values():
        # the rule becomes relevant only if redundant
        if group["occurrences"] > 1:
            group["tree_indices"] = sorted(group["tree_indices"])
            group["tree_count"] = len(group["tree_indices"])
            group["total_tree_count"] = total_tree_count
            group["forest_coverage_percentage"] = (group["tree_count"] / total_tree_count * 100) #tot. no of trees containg the rule r/tot no of trees
            frequent_rules.append(group)
    frequent_rules.sort(key=lambda group: (group["tree_count"], group["occurrences"]), reverse=True)

    logger.info("Found %d exact rules occurring more than once.", len(frequent_rules))

    return frequent_rules

def inspect_tree(model: RandomForestClassifier, feature_names: Sequence[str], tree_index: int):
    """
    Prints basic information about one decision tree inside the Random Forest
    """
    estimator = model.estimators_[tree_index]
    tree = estimator.tree_

    inspection_rows = []

    logger.info("\033[33mInspecting tree %d\033[0m", tree_index)
    logger.info("\033[32mTree node count: %d\033[0m", tree.node_count)
    logger.info("\033[32mTree max depth: %d\033[0m", estimator.get_depth())
    logger.info("\033[32mTree number of leaves: %d\033[0m", estimator.get_n_leaves())

    #max_nodes_to_print = min(10, tree.node_count)
    max_nodes_to_print = tree.node_count
    logger.info("\033[33m%d nodes to be printed:\033[0m", max_nodes_to_print)

    for node_id in range(tree.node_count):
        feature_index = tree.feature[node_id]

        if feature_index == -2: # leaf node
            samples = tree.n_node_samples[node_id]
            class_values = tree.value[node_id][0]
            predicted_class_index = int(class_values.argmax())
            predicted_class_label = model.classes_[predicted_class_index]
            inspection_rows.append({
                "tree_index": tree_index,
                "node_id": node_id,
                "node_type": "leaf",
                "split_type": "",
                "feature_index": "",
                "feature_name": "",
                "condition": "",
                "threshold": "",
                "missing_go_to_left": "",
                "left_child": "",
                "right_child": "",
                "samples": int(samples),
                "predicted_class_index": predicted_class_index,
                "predicted_class_label": str(predicted_class_label),
                "class_values": json.dumps(class_values.tolist())
            })

            logger.info(f"Node {node_id}: LEAF : samples={samples} ; predicted_class_index={predicted_class_index} ; class_values={class_values}")

        else: #intermediate node
            feature_name = feature_names[feature_index]
            threshold = float(tree.threshold[node_id])
            left_child = int(tree.children_left[node_id])
            right_child = int(tree.children_right[node_id])
            samples = int(tree.n_node_samples[node_id])
            missing_go_to_left = (bool(tree.missing_go_to_left[node_id]) if hasattr(tree, "missing_go_to_left") else None)
            split_type, threshold_for_csv, condition = _expand_condition(missing_go_to_left, threshold, feature_name,
                                                                          node_id, left_child, right_child)
            inspection_rows.append({
                "tree_index": tree_index,
                "node_id": node_id,
                "node_type": "split",
                "split_type": split_type,
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "condition": condition,
                "threshold": threshold_for_csv,
                "missing_go_to_left": missing_go_to_left,
                "left_child": left_child,
                "right_child": right_child,
                "samples": samples,
                "predicted_class_index": "",
                "predicted_class_label": "",
                "class_values": ""
            })
    return inspection_rows

def _visit_node(model, tree, feature_names, tree_index, node_id, current_conditions, rules):
    """
    Visits the node, traverses the tree and extracts rules from the current node.
    """
    feature_index = int(tree.feature[node_id])
    if feature_index == -2: #leaf node
        samples = int(tree.n_node_samples[node_id])
        class_values = tree.value[node_id][0]
        predicted_class_index = int(class_values.argmax())
        predicted_class_label = model.classes_[predicted_class_index]
        rule = {
            "tree_index": tree_index,
            "leaf_node_id": node_id,
            "conditions": list(current_conditions),
            "predicted_class_index": predicted_class_index,
            "predicted_class_label": str(predicted_class_label),
            "samples": samples,
            "class_values": class_values.tolist()
        }
        rules.append(rule)
        return
    else: #intermediate node
        feature_name = feature_names[feature_index]
        threshold = float(tree.threshold[node_id])
        left_child = int(tree.children_left[node_id])
        right_child = int(tree.children_right[node_id])
        #samples = int(tree.n_node_samples[node_id])
        missing_go_to_left = (bool(tree.missing_go_to_left[node_id]) if hasattr(tree, "missing_go_to_left") else None)
        right_accepts_missing = (None if missing_go_to_left is None else not missing_go_to_left)
        if not math.isnan(threshold):
            left_condition = {
                "feature_index": feature_index,
                "feature_name": feature_name,
                "operator": "<=",
                "threshold": threshold,
                "accepts_missing": missing_go_to_left
            }
            right_condition = {
                "feature_index": feature_index,
                "feature_name": feature_name,
                "operator": ">",
                "threshold": threshold,
                "accepts_missing": right_accepts_missing
            }
        else:
            if missing_go_to_left:
                left_operator = "IS MISSING"
                right_operator = "IS NOT MISSING"
            else:
                left_operator = "IS NOT MISSING"
                right_operator = "IS MISSING"
            left_condition = {
                "feature_index": feature_index,
                "feature_name": feature_name,
                "operator": left_operator,
                "threshold": None,
                "accepts_missing": missing_go_to_left
            }
            right_condition = {
                "feature_index": feature_index,
                "feature_name": feature_name,
                "operator": right_operator,
                "threshold": None,
                "accepts_missing": right_accepts_missing
            }
        left_condition["node_id"] = node_id
        left_condition["branch"] = "left"
        left_condition["child_node_id"] = left_child

        right_condition["node_id"] = node_id
        right_condition["branch"] = "right"
        right_condition["child_node_id"] = right_child

        left_conditions = current_conditions + [left_condition]
        right_conditions = current_conditions + [right_condition]
        _visit_node(model, tree, feature_names, tree_index, left_child, left_conditions, rules)
        _visit_node(model, tree, feature_names, tree_index, right_child, right_conditions, rules)
        return

def _format_condition(condition, use_original_threshold=False):
    """
    Formats a condition. If use_original_threshold=False, then scalers are used.
    """
    feature_name = condition["feature_name"]
    operator = condition["operator"]
    threshold = condition["threshold"]
    accepts_missing = condition["accepts_missing"]

    if operator in {"IS MISSING", "IS NOT MISSING"}:
        return f"{feature_name} {operator}"

    if use_original_threshold:
        threshold = condition.get("threshold_original")
        if threshold is None:
            raise ValueError(f"Original threshold is unavailable for feature {feature_name}.")
    else:
        threshold = condition["threshold"]

    numeric_condition = (f"{feature_name} {operator} {threshold:.6f}")

    if accepts_missing is True:
        return (f"({numeric_condition} OR {feature_name} IS MISSING)")

    """
    if accepts_missing is False:
        return (f"({feature_name} IS NOT MISSING AND {numeric_condition})")
    """

    return numeric_condition

def _format_rule(rule, use_original_threshold=False):
    """
    Converts the conditions into a text and adds the text to a list.
    All conditions are then joined through an 'AND'.
    Each rule is finally formatted as 'IF conditions THEN class = outcome'.
    """
    rule_text = "IF "
    for condition_index, condition in enumerate(rule["conditions"]):
        condition_text = _format_condition(condition, use_original_threshold)
        if condition_index > 0:
            rule_text += " AND "

        rule_text += condition_text

    if not rule["conditions"]:
        rule_text += "TRUE"

    outcome = rule["predicted_class_label"]
    #outcome = 'licit' if outcome==0 else 'suspicious'
    rule_text += f" THEN class = {outcome}"

    return rule_text

def _validate_rule_path(tree, rule):
    expected_node_id = 0

    for condition in rule["conditions"]:
        node_id = condition["node_id"]
        branch = condition["branch"]
        child_node_id = condition["child_node_id"]

        if node_id != expected_node_id:
            raise RuntimeError(f"Broken rule path: expected node {expected_node_id} found node {node_id}.")

        if branch == "left":
            expected_child = int(tree.children_left[node_id])
        elif branch == "right":
            expected_child = int(tree.children_right[node_id])
        else:
            raise RuntimeError(f"Unknown branch: {branch}")

        if child_node_id != expected_child:
            raise RuntimeError(f"Invalid child for node {node_id}: expected {expected_child}, found {child_node_id}.")

        expected_node_id = child_node_id
    if expected_node_id != rule["leaf_node_id"]:
        raise RuntimeError(f"Rule ends at node {expected_node_id}, but its declared leaf is {rule['leaf_node_id']}.")

def _recover_original_threshold(condition, scalers):
    threshold = condition["threshold"]
    if threshold is None:
        return None

    feature_index = condition["feature_index"]
    scaler = scalers[feature_index]
    if not isinstance(scaler, StandardScaler):
        return None
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError(f"StandardScaler for feature {feature_index} does not appear to be fitted.")

    mean = float(scaler.mean_[0])
    scale = float(scaler.scale_[0])
    original_threshold = threshold * scale + mean

    return float(original_threshold)

def add_original_thresholds(rules, scalers):
    """
    Adds up the original threshold to all the conditions. This works only when the --scalers argument is passed.
    For each rule -> condition, it computes the original threshold and adds up the threshold_original
    """
    converted_thresholds = 0

    for rule in rules:
        for condition in rule["conditions"]:
            original_threshold = _recover_original_threshold(condition=condition, scalers=scalers)
            condition["threshold_original"] = original_threshold
            if original_threshold is not None:
                converted_thresholds += 1

    logger.info(f"Recovered {converted_thresholds} original-scale thresholds.")

def extract_rules(model: RandomForestClassifier, feature_names: Sequence[str], tree_index: int):
    """
    Extracts rules from a single dicision tree in the Random Forest.
    """
    estimator = model.estimators_[tree_index]
    tree = estimator.tree_

    node_count = tree.node_count
    logger.info("\033[33mExtracting rules from tree %d\033[0m", tree_index)
    logger.info("\033[32mTree node count: %d\033[0m", node_count)
    logger.info("\033[32mTree max depth: %d\033[0m", estimator.get_depth())
    logger.info("\033[32mTree number of leaves/rules: %d\033[0m", estimator.get_n_leaves())

    rules = []
    current_conditions = []
    _visit_node(model, tree, feature_names, tree_index, node_id=0, current_conditions=current_conditions, rules= rules)
    for rule in rules:
        _validate_rule_path(tree, rule)

    expected_rules = estimator.get_n_leaves()
    if len(rules) != expected_rules:
        raise RuntimeError(f"Rule extraction inconsistency: expected {expected_rules} rules, but extracted {len(rules)}.")
    logger.info(f"Successfully extracted {len(rules)} rules from tree {tree_index}")
    return rules

def load_rules_json(rules_path: Path):
    """
    Loads previously extracted rules from a .json file.
    """
    logger.info("Loading rules from %s", rules_path)

    with rules_path.open(mode="r", encoding="utf-8") as file:
        rules = json.load(file)

    if not isinstance(rules, list):
        raise TypeError("The rules.json file does not contain a list of rules.")

    logger.info("Successfully loaded %d rules.",len(rules))
    return rules

def load_model(model_path):
    """
    Loads the pre-trained Random Forest Classifier's model from a pickle file.
    Checks whether the model it is a RandomForestClassifier instance or not.
    """
    logger.info("\033[33mLoading model from %s\033[0m", model_path)
    with model_path.open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(f"Expected a RandomForestClassifier, found {type(model).__name__}.")
    return model

def load_feature_names(features_path: Path):
    """
    Loads feature names in the correct order.
    """
    logger.info("\033[33mLoading feature names from %s\033[0m", features_path)

    feature_names = []
    with features_path.open(mode="r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_columns = {"feature_position", "feature_name"}

        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Feature file is missing required columns: {sorted(missing_columns)}")

        rows = []
        for row in reader:
            rows.append({
                "feature_position": int(row["feature_position"]),
                "feature_name": row["feature_name"]
            })
    rows.sort(key=lambda row: row["feature_position"])

    expected_positions = list(range(len(rows)))

    actual_position = []
    for row in rows:
        actual_position.append(row["feature_position"])

    if actual_position != expected_positions:
        raise ValueError(f"Feature positions are not contiguous or do not start from 0. Expected {expected_positions[:10]}..., found {actual_position[:10]}...")

    feature_names = []
    for row in rows:
        feature_name = row["feature_name"]
        feature_names.append(feature_name)

    return feature_names

def save_inspection_csv(inspection_rows: list[dict], output_path: Path):
    """
    Save inspected tree nodes to CSV
    """
    if not inspection_rows:
        raise ValueError("No inspection rows are available to be saved.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "tree_index",
        "node_id",
        "node_type",
        "split_type",
        "feature_index",
        "feature_name",
        "condition",
        "threshold",
        "missing_go_to_left",
        "left_child",
        "right_child",
        "samples",
        "predicted_class_index",
        "predicted_class_label",
        "class_values"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inspection_rows)

    logger.info("\033[33mTree inspection saved to %s\033[0m", output_path)

def save_tree_visualization( model: RandomForestClassifier, feature_names: Sequence[str], tree_index: int, output_dir: Path, max_depth: int):
    """
    Saves one decision tree as a Graphviz .dot file and, when Gtaphviz is available, also extracts its .png.
    """
    estimator = model.estimators_[tree_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_path = output_dir / f"tree_{tree_index:03d}.dot"
    png_path = output_dir / f"tree_{tree_index:03d}.png"

    export_graphviz(
        # the tree to export
        estimator,
        # output directory for the .dot file
        out_file=str(dot_path),
        # names of the features
        feature_names=list(feature_names),
        # class label names
        class_names=[str(label) for label in model.classes_],
        # changes nodes' colour
        filled=True,
        # roundes nodes visualization
        rounded=True,
        # special characters cannot be used
        special_characters=False,
        # impurity level of the node (e.g., gini impurity)
        impurity=True,
        # if False, no percentage but numbers (e.g. samples = 3 instead of 3.3% and value=[6, 48] instead of [0.1111, 0.88889]
        proportion=False,
        # number of decimals in floating numbers
        precision=4,
        # depth of the visualization
        max_depth=max_depth,
        # shows the id of the node in the label
        node_ids=True
    )
    replace_missing_split_labels(dot_path=dot_path, estimator=estimator, feature_names=feature_names)
    logger.info(f"\033[33mTree {tree_index} .dot visualization saved to {dot_path}\033[0m")

    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"\033[33mTree {tree_index} .png graph saved to {png_path}\033[0m")

    except FileNotFoundError:
        logger.warning(f"Graphviz executable 'dot' is not installed or not available in the path. The .dot file was still generated: {dot_path}")

    except subprocess.CalledProcessError as exc:
        logger.warning(f"Graphviz failed while graphically creating the tree {tree_index}. The .dot file is still available at {dot_path}. Error: {exc.stderr}")

def save_frequent_rules_json(frequent_rules, output_path: Path):
    """
    Saves frequent-rule groups in JSON format.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_rules = []

    for rank, group in enumerate(frequent_rules, start=1):
        representative_rule = group["representative_rule"]
        output_rules.append({
            "frequency_rank": rank, # posizione in graduatoria della regola nella graduatoria delle regole frequenti
            "occurrences": group["occurrences"],
            "matches": group["matches"],
            "tree_count": group["tree_count"],
            "total_tree_count": group["total_tree_count"],
            "forest_coverage_percentage": (group["forest_coverage_percentage"]), # percentage of trees using that rule
            #"support_percentage": group["percentage"] * 100,
            "tree_indices": group["tree_indices"],
            "condition_count": len(representative_rule["conditions"]),
            "predicted_class_index": representative_rule["predicted_class_index"],
            "predicted_class_label": representative_rule.get("predicted_class_label"),
            "conditions": representative_rule["conditions"],
            "formatted_rule": _format_rule(representative_rule, use_original_threshold=False)
        })
    with output_path.open(mode="w", encoding="utf-8") as file:
        json.dump(output_rules, file, indent=2, ensure_ascii=False)

    logger.info(f"Frequent rules saved to {output_path}")

def save_frequent_rules_summary_csv(frequent_rules, output_path: Path):
    """
    Saves one summary row for each frequent-rule group.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frequency_rank",
        "occurrences",
        "tree_count",
        "total_tree_count",
        "forest_coverage_percentage",
        "condition_count",
        "predicted_class",
        "tree_indices",
        "source_rule_ids",
        "matches",
        "formatted_rule"
    ]

    with output_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for frequency_rank, group in enumerate(frequent_rules, start=1):
            representative_rule = group["representative_rule"]

            writer.writerow({
                "frequency_rank": frequency_rank,
                "occurrences": group["occurrences"],
                "tree_count": group["tree_count"],
                "total_tree_count": group["total_tree_count"],
                "forest_coverage_percentage": (group["forest_coverage_percentage"]),
                "condition_count": len(representative_rule["conditions"]),
                "predicted_class": (representative_rule["predicted_class_label"]),
                "tree_indices": json.dumps(group["tree_indices"]), # dumps stores the list into a single cell of the csv
                "source_rule_ids": json.dumps([match["source_rule_id"] for match in group["matches"]]),
                "matches": json.dumps(group["matches"]),
                "formatted_rule": _format_rule(representative_rule, use_original_threshold=False)
            })

    logger.info("Frequent-rule summary saved to %s", output_path)


def analyze_and_save_frequent_rules(rules, output_dir):
    """
    Finds frequent rules and saves them.
    """
    validate_rules(rules)

    logger.info("Analysing %d extracted rules.", len(rules))

    frequent_rules = find_frequent_rules(rules, ignore_condition_order=True)
    if not frequent_rules:
        logger.info("No duplicate rules were found.")
        return []

    first_group = frequent_rules[0]
    representative_rule = first_group["representative_rule"]
    logger.info(f"Most frequent exact rule: occurrences={first_group['occurrences']}, tree_count={first_group['tree_count']}, forest_coverage=%.2f%%, trees={first_group['tree_indices']}",  first_group["forest_coverage_percentage"])
    logger.info(f"Most frequent rule has {len(representative_rule['conditions'])} conditions and predicts class {representative_rule['predicted_class_index']}.")
    logger.info("Most frequent rule:\n%s", _format_rule(representative_rule, use_original_threshold=False))

    frequent_json_path = (output_dir / "frequent_rules.json")
    frequent_txt_path = (output_dir / "frequent_rules.txt")
    frequent_csv_path = (output_dir / "frequent_rules_summary.csv")
    save_frequent_rules_json(frequent_rules=frequent_rules, output_path=frequent_json_path)
    save_frequent_rules_txt(frequent_rules=frequent_rules, output_path=frequent_txt_path)
    save_frequent_rules_summary_csv(frequent_rules=frequent_rules, output_path=frequent_csv_path)
    return frequent_rules

def save_frequent_rules_txt(frequent_rules, output_path: Path):
    """
    Saves frequent rules in a human-readable format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", encoding="utf-8") as file:

        for frequency_rank, group in enumerate(frequent_rules, start=1):
            representative_rule = group["representative_rule"]

            formatted_rule = _format_rule(representative_rule, use_original_threshold=False)

            file.write(f"FREQUENT RULE {frequency_rank}\n")
            file.write(f"Occurrences: {group['occurrences']}\n")
            file.write(f"Trees containing the rule: {group['tree_count']}\n")
            file.write("Matches:\n")
            for match in group["matches"]:
                file.write(f"  - tree {match['tree_index']}, leaf {match['leaf_node_id']}\n")
            #file.write(f"Percentage of trees using the rule in the forest: {group['percentage'] * 100:.2f}%\n")
            file.write(f"Forest coverage: {group['forest_coverage_percentage']:.2f}%\n")
            file.write(f"Tree indices: {group['tree_indices']}\n")
            file.write(f"Predicted class: {representative_rule['predicted_class_label']}\n")
            file.write(f"Number of conditions: {len(representative_rule['conditions'])}\n")
            file.write(f"{formatted_rule}\n\n")
    logger.info(f"Human-readable frequent rules saved to {output_path}")

def replace_missing_split_labels(dot_path: Path, estimator, feature_names: Sequence[str]):
    """
    Replace Graphviz labels including nan (e.g., 'feature <= nan') with explicit
    missing-value conditions (e.g., 'feature IS MISSING' or 'IS NOT MISSING').
    The current function does not modify tree/estimator's behaviour it just
    adjusts the Graphvix .dot text file so to let nodes, separating missing values, become meaningful.
    """
    # recover tree's internal structure
    tree = estimator.tree_
    #read dot file
    dot_text = dot_path.read_text(encoding="utf-8")

    # check whether the current tree includes missing_go_to_left values
    if not hasattr(tree, "missing_go_to_left"):
        logger.warning("The current tree does not include missing_go_to_left. NaN split labels cannot be rewritten safely.")
        return

    replaced_nodes = 0
    # check for nan values
    for node_id, threshold in enumerate(tree.threshold):
        # ignore normal numerical splits
        if not math.isnan(float(threshold)):
            # no nan? continue to next node
            continue

        feature_index = int(tree.feature[node_id])

        if feature_index < 0:
            continue

        feature_name = feature_names[feature_index]
        missing_go_to_left = bool(tree.missing_go_to_left[node_id])

        original_condition = f"{feature_name} <= nan"

        if missing_go_to_left:
            readable_condition = f"{feature_name} IS MISSING"
        else:
            readable_condition = f"{feature_name} IS NOT MISSING"

        if original_condition not in dot_text:
            logger.warning(f"Could not find Graphviz condition for tree node {node_id}: {original_condition}")
            continue

        dot_text = dot_text.replace(original_condition, readable_condition, 1)

        replaced_nodes += 1

        logger.info(f"Tree node {node_id} missing split rewritten as: {readable_condition}")

    dot_path.write_text(dot_text, encoding="utf-8")

    logger.info(f"\033[33mRewritten {replaced_nodes} missing-value split label(s) in {dot_path}\033[0m")

def save_rules_txt(formatted_rules, output_path: Path):
    if not formatted_rules:
        raise ValueError("No formatted rules are available to be saved.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for rule_index, rule_text in enumerate(formatted_rules, start=1):
            file.write(f"RULE {rule_index}\n")
            file.write(rule_text)
            file.write("\n\n")

    logger.info(f"Formatted rules saved to {output_path}")

def save_rules_json(rules, output_path: Path):
    if not rules:
        raise ValueError("No rules are available to be saved.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        # directly writes on file (does not produce a string)
        json.dump(
            rules,
            file,
            indent=2, # more readable
            ensure_ascii=False #preserves non ASCII chars
        )

    logger.info("Rules saved to %s", output_path)

def save_rules_summary_csv(rules, output_path: Path):
    if not rules:
        raise ValueError("No rules are available to be summarized.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rule_id",
        "tree_index",
        "leaf_node_id",
        "depth",
        "predicted_class",
        "tree_train_samples"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rule_index, rule in enumerate(rules, start=1):
            writer.writerow({
                "rule_id": rule_index,
                "tree_index": rule["tree_index"],
                "leaf_node_id": rule["leaf_node_id"],
                "depth": len(rule["conditions"]), # tot. no of conditions in the path
                "predicted_class": rule["predicted_class_label"],
                "tree_train_samples": rule["samples"] # tot. no of training samples that reached the leaf
            })
    logger.info(f"Rule summary saved to {output_path}")

def main():
    logger.info("\033[94mInitializing 3REx\033[0m")
    # read arguments
    args = parse_arguments()
    if args.max_trees < 1:
        raise ValueError("--max-trees must be greater than or equal to 1.")
    if args.visualization_depth < 1:
        raise ValueError("--visualization-depth must be greater than or equal to 1.")
    model_path = args.model
    features_path = args.features
    scalers_path = args.scalers
    rules_path = args.rules
    if args.mode == "frequent-rules":
        if rules_path is None:
            raise ValueError("--rules is required when --mode frequent-rules is selected.")
        validate_input(rules_path, "Rules file")
        rules = load_rules_json(rules_path)
        validate_rules(rules)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = (args.output_dir / "frequent_rules" / timestamp)
        analyze_and_save_frequent_rules(rules=rules, output_dir=run_output_dir)
        return

    if model_path is None:
        raise ValueError(f"--model is required when --mode {args.mode} is selected.")
    # control file existance
    validate_input(model_path, "Model file")
    if features_path is not None:
        validate_input(features_path, "Feature file")
    if scalers_path is not None:
        validate_input(scalers_path, "Scalers file")
    # load model
    model = load_model(model_path)
    scalers = None
    if scalers_path is not None:
        scalers = load_scalers(scalers_path)
        validate_model_scalers(model=model, scalers=scalers)
        logger.info(f"Successfully loaded and validated {len(scalers)} preprocessing objects.")

    # if the feature_names csv exists, recover feature names
    if features_path is not None:
        feature_names = load_feature_names(features_path)
    # if the user did not provide the csv, try to extract feature names from the model
    elif hasattr(model, "feature_names_in_"):
        feature_names = []
        for feature_name in model.feature_names_in_:
            feature_names.append(str(feature_name))
        logger.info("Feature names recovered from the model.")
    # if no feature name is available, automatically generate them
    else:
        feature_names = []
        for feature_index in range(model.n_features_in_):
            feature_names.append(f"feature_{feature_index}")

        logger.warning("No feature names were provided or stored in the model. Generated generic feature names.")

    # verify that model features' number == #features in features_names.csv
    validate_model_features(model=model, feature_names=feature_names)

    logger.info("\033[94mInput validation completed successfully.\033[0m")
    logger.info("\033[32mModel type: %s\033[0m", type(model).__name__)
    logger.info("\033[32mNumber of trees: %d\033[0m", len(model.estimators_))
    logger.info("\033[32mNumber of features: %d\033[0m", len(feature_names))
    logger.info("\033[32mFirst feature: %s\033[0m", feature_names[0])
    logger.info("\033[32mLast feature: %s\033[0m", feature_names[-1])
    if args.mode == "inspect":
        total_trees = len(model.estimators_)
        trees_to_inspect = min(args.max_trees, total_trees)
        logger.info("\033[33mInspecting %d tree(s) out of %d\033[0m", trees_to_inspect, total_trees)

        inspection_rows = []

        for tree_index in range(trees_to_inspect):
            tree_rows = inspect_tree(model=model, feature_names=feature_names, tree_index=tree_index)
            inspection_rows.extend(tree_rows)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_path.stem

        run_output_dir = (args.output_dir / model_name / f"tree_inspection_{timestamp}")

        inspection_output_path = run_output_dir / "tree_inspection.csv"

        save_inspection_csv(inspection_rows=inspection_rows, output_path=inspection_output_path)
        for tree_index in range(trees_to_inspect):
            save_tree_visualization( model=model, feature_names=feature_names, tree_index=tree_index, output_dir=run_output_dir, max_depth=args.visualization_depth)
    if args.mode == "extract":
        total_trees = len(model.estimators_)
        logger.info(f"\033[33mExtracting rules from {total_trees} tree(s).\033[0m")

        all_rules = []

        for tree_index in range(total_trees):
            tree_rows = extract_rules(model=model, feature_names=feature_names, tree_index=tree_index)
            all_rules.extend(tree_rows) #extend e non append per evitare di creare lista di liste
        """
        if all_rules:
            logger.info("\nFirst extracted rule:\n%s", json.dumps(all_rules[0], indent=2))
            logger.info(f"\nFirst formatted rule:\n{_format_rule(all_rules[0])}")
        

        if scalers is not None and all_rules:
            first_condition = all_rules[0]["conditions"][0]

            original_threshold = _recover_original_threshold(condition=first_condition, scalers=scalers)

            logger.info(f"First condition model threshold: {first_condition['threshold']}")
            logger.info(f"First condition original threshold: {original_threshold}")
        """
        for rule_id, rule in enumerate(all_rules, start=1):
            rule["rule_id"] = rule_id
        if scalers is not None and all_rules:
            add_original_thresholds(rules=all_rules, scalers=scalers)
            scaler_type_counts = {}
            for scaler in scalers:
                scaler_type = type(scaler).__name__
                if scaler_type not in scaler_type_counts:
                    scaler_type_counts[scaler_type] = 0
                scaler_type_counts[scaler_type] += 1
            logger.info(f"Object types: {scaler_type_counts}")

        formatted_rules_model = []
        formatted_rules_original = []
        for rule in all_rules:
            model_rule = _format_rule(rule, use_original_threshold=False)
            formatted_rules_model.append(model_rule)
            if scalers is not None:
                original_rule = _format_rule(rule, use_original_threshold=True)
                formatted_rules_original.append(original_rule)
        if len(formatted_rules_model) != len(all_rules):
            raise RuntimeError(f"Rule formatting inconsistency: extracted {len(all_rules)} rules, but formatted {len(formatted_rules_model)}.")
        if scalers is not None and len(formatted_rules_original) != len(all_rules):
            raise RuntimeError(f"Scaler-space inconsistency: extracted {len(all_rules)} rules, but formatted {len(formatted_rules_original)}.")
        if formatted_rules_model:
            logger.info(f"First formatted rule:\n{formatted_rules_model[0]}")
        logger.info(f"\033[32mSuccessfully extracted {len(all_rules)} total rules.\033[0m")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_path.stem

        run_output_dir = (args.output_dir / model_name / f"rules_extraction_{timestamp}")

        txt_output_path = run_output_dir / "rules.txt"

        json_output_path = run_output_dir / "rules.json"
        summary_output_path = (run_output_dir / "rules_summary.csv")

        save_rules_txt(formatted_rules=formatted_rules_model, output_path=txt_output_path)
        if scalers is not None:
            txt_output_path_scalers = run_output_dir / "rules_with_scalers.txt"
            save_rules_txt(formatted_rules=formatted_rules_original, output_path=txt_output_path_scalers)

        save_rules_json(rules=all_rules, output_path=json_output_path)
        save_rules_summary_csv(rules=all_rules, output_path=summary_output_path)
        frequent_rules_output_dir = (run_output_dir / "frequent_rules")
        analyze_and_save_frequent_rules(rules=all_rules, output_dir=frequent_rules_output_dir)



if __name__ == "__main__":
    start_time = time.time()

    try:
        main()
    except Exception:
        logger.exception("3REx execution failed.")
        raise
    finally:
        elapsed_seconds = time.time() - start_time
        logger.info("\033[33mTotal runtime: %.2fs\033[0m", elapsed_seconds)
        logger.info("\033[33mTotal runtime: %.2fmin\033[0m", elapsed_seconds / 60)
