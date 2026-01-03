import json
from jsonschema import Draft202012Validator
from collections import defaultdict
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def get_errors_dict(validator, comparison_json):
    """Generate dict of {error type: path to error} for JSON"""
    errors_dict = defaultdict(list)
    for error in validator.iter_errors(comparison_json):
        errors_dict[error.validator].append(list(error.path))
    return errors_dict


def calculate_validity_score(errors_dict):
    """
    Produce validity score for a JSON object for prompt evaluation.
    Gives more negative weight to errors at higher depths
    and different weights to different types of errors
    """
    severity_weights = {
        'required': 10,  # High penalty for missing required properties
        'type': 5,  # Medium penalty for type mismatches
        'enum': 2.5,  # Low penalty for enum violations
    }

    negative_marking = 0
    error_details = []

    for validator, paths in errors_dict.items():
        weight = severity_weights.get(validator, 3)  # Default weight if validator not in map

        for path in paths:
            depth = len(path)
            if validator == 'required' and depth == 0:
                adjusted_weight = 50  # Maximum penalty for missing required property at root depth
            else:
                adj_depth = depth - 1 if depth > 0 else depth  # depth weighting starts at one level below root
                adjusted_weight = weight * (0.8 ** adj_depth)  # Reduce weight by 20% per depth level
            negative_marking += adjusted_weight
            error_details.append({
                'validator': validator,
                'path': path,
                'penalty': round(adjusted_weight, 2),
                'depth': depth
            })

    # 0 <= score <= 100
    validity_score = 100 - max(negative_marking, 0) if negative_marking < 100 else 0

    return validity_score, error_details


def generate_analysis_log(errors_dict, validity_score, error_details):
    """Build visual output of JSON validity analysis"""
    total_errors = sum(len(paths) for paths in errors_dict.values())

    # first visual element: displays summary stats
    summary_text = Text.assemble(
        ("Overall Validity Score: ", "default"),
        (f"{validity_score:.2f}/100", "bold yellow"),
        ("\nTotal Errors: ", "default"),
        (f"{total_errors}", "bold red" if total_errors > 0 else "bold green")
    )
    summary_panel = Panel(
        summary_text,
        title="[bold]Validation Summary[/bold]",
        border_style="blue",
        expand=False
    )

    if error_details:
        # second visual element: detailed error breakdown
        # includes path to error, error type, penalty, and depth for each error
        error_table = Table(
            title="[bold]Detailed Error Breakdown[/bold]",
            show_header=True,
            header_style="bold magenta",
            box=None
        )
        error_table.add_column("Field Path", style="cyan", no_wrap=True)
        error_table.add_column("Violation Type", style="red")
        error_table.add_column("Penalty", justify="right", style="yellow")
        error_table.add_column("Depth", justify="right", style="dim blue")

        for error in error_details:
            path = error.get('path')
            error_table.add_row(
                '.'.join(map(str, path)) if path else 'root',  # path string with '.' separating each element in the path name
                f"{error.get('validator', '?')}",
                f"{error.get('penalty', 0.0):.1f}",
                f"{error.get('depth', '?')}"
            )
        error_details_element = error_table
    else:
        error_details_element = "[bold green]No validation errors found.[/bold green]"

    return summary_panel, error_details_element


def analyze_json(validator, comparison_json):
    errors_dict = get_errors_dict(validator, comparison_json)
    validity_score, error_details = calculate_validity_score(errors_dict)
    json_analysis_log = generate_analysis_log(errors_dict, validity_score, error_details)
    return validity_score, errors_dict, json_analysis_log


def compute_prompt_performance(all_jsons, validator):
    """Determine performance of a prompt across all LLMs"""
    score_sum = 0
    max_score = [0, 0]
    all_logs = []
    num_incorrect_responses = 0

    for i, json_data in enumerate(all_jsons):
        if not isinstance(json_data, dict):
            all_logs.append('No valid JSON provided in LLM response.')
            num_incorrect_responses += 1
            continue

        validity_score, errors_dict, json_analysis_log = analyze_json(validator, json_data)
        score_sum += validity_score
        max_score[1] = i if validity_score > max_score[0] else max_score[1]
        max_score[0] = max(max_score[0], validity_score)
        all_logs.append(json_analysis_log)

    return score_sum / len(all_jsons), max_score, all_logs, num_incorrect_responses


def count_all_fields(data):
    """Count all fields (including nested) in a JSON object."""
    count = 0
    if isinstance(data, dict):
        for key, value in data.items():
            count += 1
            count += count_all_fields(value)
    return count


def calculate_aggregate_metrics(all_jsons, validator):
    """
    Defining precision and recall to compute F1 score to evaluate an LLM:

    True Positives (TP): Fields that are correctly present and valid according to the schema.
    False Positives (FP): Fields that are present but invalid (e.g., wrong type, enum violation) or extra fields not in the schema.
    False Negatives (FN): Required fields missing from the JSON.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for json_data in all_jsons:
        if not isinstance(json_data, dict):
            continue

        errors_dict = get_errors_dict(validator, json_data)

        # Count required errors (FN)
        fn = sum(len(paths) for val_type, paths in errors_dict.items()
                 if val_type == 'required')
        total_fn += fn

        # Count additionalProperties errors (extra fields)
        extra_fields = sum(len(paths) for val_type, paths in errors_dict.items()
                           if val_type == 'additionalProperties')

        # Count type/enum errors (invalid fields)
        invalid_fields = sum(len(paths) for val_type, paths in errors_dict.items()
                             if val_type not in ['required', 'additionalProperties'])

        # Calculate TP/FP for this JSON
        total_fields = count_all_fields(json_data)
        tp = (total_fields - extra_fields) - invalid_fields
        fp = extra_fields + invalid_fields

        total_tp += tp
        total_fp += fp

    return total_tp, total_fp, total_fn


def compute_llm_performance(all_jsons, validator):
    """Determine performance of an LLM across all prompts"""
    tp, fp, fn = calculate_aggregate_metrics(all_jsons, validator)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


config_schema = load_json('database_config_schema.json')
config_validator = Draft202012Validator(config_schema)
