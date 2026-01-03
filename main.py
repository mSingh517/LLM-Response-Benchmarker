import os
import pandas as pd
import typer
from typing_extensions import Annotated
from rich import print
from rich.console import Console
from rich.table import Table
import llm_config_generator as generator
import llm_benchmarker as benchmarker
from llm_prompt_utils import llms, prompts

keys = [
    os.environ.get('GPT_KEY'),
    os.environ.get('CLAUDE_KEY'),
    os.environ.get('GEMINI_KEY'),
    os.environ.get('MISTRAL_KEY')
]

app = typer.Typer(
    help='An LLM-driven Database Connector Config JSON Generator and Benchmarker\n\n'
         'You must create the following API Key environment variables '
         'and enter their respective values to access the LLMs:\n\n'
         'GPT_KEY\n\nCLAUDE_KEY\n\nGEMINI_KEY\n\nMISTRAL_KEY'
)

command_categories = [
    'Show existing data',
    'Generate new data (Requires API Keys)'
]

console = Console()


def arg_max(nums):
    return max(enumerate(nums), key=lambda x: x[1])


def print_prompt_performance(avg_score, max_score, all_logs, num_incorrect, prompt_num, see_logs):
    """Build visual output of prompt performance metrics"""
    table = Table(title=f'Prompt {prompt_num} Performance Summary', title_style="bold dark_green")

    table.add_column('Avg Validity Score', justify="center", style="cyan", header_style="bold cyan")
    table.add_column('Max Validity Score', justify="center", style="magenta", header_style="bold light_coral")
    table.add_column('Unparseable responses', justify="center", style="red", header_style="bold slate_blue1")

    avg_score_str = f'[bold cyan]{avg_score:.2f}/100.00[/bold cyan]'
    max_score_str = f'[bold][light_coral]{max_score[0]:.2f}/100.00[/bold]\n\nachieved by[/light_coral] [bold yellow]{llms[max_score[1]]}[/bold yellow]'
    num_incorrect_str = f'[bold slate_blue1]{num_incorrect}[/bold slate_blue1]'

    table.add_row(
        avg_score_str,
        max_score_str,
        num_incorrect_str
    )
    print('[bold]Here is the overall performance of the prompt[/bold]:\n')
    console.print(table)

    if see_logs:
        print('\n[bold]Here are the detailed error logs for each LLM:[/bold]')
        for i, log_out in enumerate(all_logs):
            console.print(f'\n[bold dark_orange]{llms[i]}[/bold dark_orange]')
            if not isinstance(log_out, str):
                for val in log_out:
                    console.print(val)
            else:
                console.print(log_out)


def print_model_performance(model, precision, recall, f1):
    """Build visual output of LLM performance metrics"""
    table = Table(title=f'{model} Performance', title_style="bold blue")

    table.add_column('Precision', justify="center", style="cyan", header_style="bold cyan")
    table.add_column('Recall', justify="center", style="magenta", header_style="bold light_coral")
    table.add_column('F1-Score', justify="center", style="green", header_style="bold dark_sea_green4")

    precision_str = f'[bold cyan]{precision:.2f}[/bold cyan]'
    recall_str = f'[bold light_coral]{recall:.2f}[/bold light_coral]'
    f1_str = f'[bold dark_sea_green4]{f1:.2f}[/bold dark_sea_green4]'

    table.add_row(precision_str, recall_str, f1_str)
    console.print(table)


@app.command('display-prompts', rich_help_panel=command_categories[0])
def print_prompts():
    """Display the prompts"""
    print('[bold]Here are the prompts:[/bold]')
    for i, prompt in enumerate(prompts):
        print(f'\n[bold red3]PROMPT NUMBER {i + 1}[/bold red3]\n{prompt}\n')


@app.command('show-prompt-perf', rich_help_panel=command_categories[0])
def show_prompt_on_all_llms(
        prompt_num: Annotated[int, typer.Argument(help='The index of the prompt to show results for (1 - 6)')],
        filename: Annotated[str, typer.Argument(
            help='Name of file where existing data lies (include .pkl extension)'
        )] = 'all_data_df.pkl'
):
    """
    Display the performance of a selected prompt on all LLMs from the stored data
    """
    configs = pd.read_pickle(filename).iloc[prompt_num - 1]
    performance_metrics = benchmarker.compute_prompt_performance(configs, benchmarker.config_validator)
    print_prompt_performance(*performance_metrics, prompt_num, see_logs=True)


@app.command('show-llm-perf', rich_help_panel=command_categories[0])
def show_all_prompt_one_llm(
        model: Annotated[str, typer.Argument(
            help='Name of the model to show results for (ChatGPT, Claude, Gemini, Mistral)'
        )],
        filename: Annotated[str, typer.Argument(
            help='Name of file where existing data lies (include .pkl extension)'
        )] = 'all_data_df.pkl'
):
    """
    Display the performance of a selected LLM on all the prompts from the stored data
    """
    configs = pd.read_pickle(filename)[model]
    performance_metrics = benchmarker.compute_llm_performance(configs, benchmarker.config_validator)
    print_model_performance(model, *performance_metrics)


@app.command('show-all-perf', rich_help_panel=command_categories[0])
def see_all_existing_data_performance(
        filename: Annotated[str, typer.Argument(
            help='Name of file where existing data lies (include .pkl extension)'
        )] = 'all_data_df.pkl'
):
    """
    Display the summarized performance metrics for the stored data (all prompts on all LLMs)
    """
    loaded = pd.read_pickle(filename)
    all_avgs = []
    print('[bold]Here are the condensed performance results for each prompt across all LLMs:[/bold]')
    for i in range(len(prompts)):
        configs = loaded.iloc[i]
        performance_metrics = benchmarker.compute_prompt_performance(configs, benchmarker.config_validator)
        all_avgs.append(performance_metrics[0])
        print()
        print_prompt_performance(*performance_metrics, prompt_num=i + 1, see_logs=False)

    max_avg_index, max_avg = arg_max(all_avgs)
    print(
        f'\n[bold]Highest Average Validity Score: {max_avg:.2f}/100.00 achieved by [yellow]Prompt {max_avg_index + 1}[/yellow][/bold]'
    )

    all_f1 = []
    print('\n[bold]Here are the performance results for each LLM across all prompts:[/bold]')
    for llm in llms:
        configs = loaded[llm]
        performance_metrics = benchmarker.compute_llm_performance(configs, benchmarker.config_validator)
        all_f1.append(performance_metrics[-1])
        print()
        print_model_performance(llm, *performance_metrics)

    max_f1_index, max_f1 = arg_max(all_f1)
    print(f'\n[bold]Highest F1-Score: {max_f1:.2f} achieved by [yellow]{llms[max_f1_index]}[/yellow][/bold]')


@app.command('run-prompt', rich_help_panel=command_categories[1])
def run_prompt_on_all_llms(
        prompt_num: Annotated[int, typer.Argument(help='The index of the prompt to run (1 - 6)')],
        show: Annotated[
            bool,
            typer.Option(
                prompt='Would you like to see the performance of this prompt?',
                help='Show performance analysis without confirmation'
            )
        ]
):
    """
    Run a selected prompt on all the LLMs

    if --show is not used, will ask for confirmation
    """
    configs = generator.generate_configs_one_prompt(api_keys=keys, prompt=prompts[prompt_num - 1])
    performance_metrics = benchmarker.compute_prompt_performance(configs, benchmarker.config_validator)
    if show:
        print_prompt_performance(*performance_metrics, prompt_num, see_logs=True)


@app.command('run-llm', rich_help_panel=command_categories[1])
def run_all_prompts_one_llm(
        model: Annotated[str, typer.Argument(
            help='Name of the model to run (ChatGPT, Claude, Gemini, Mistral)'
        )],
        show: Annotated[
            bool,
            typer.Option(
                prompt='Would you like to see the performance of this LLM?',
                help='Show performance analysis without confirmation'
            )
        ]
):
    """
    Run all the prompts on a selected LLM

    if --show is not used, will ask for confirmation
    """
    api_key = keys[llms.index(model)]
    configs = generator.generate_configs_one_llm(model, api_key, prompts)
    performance_metrics = benchmarker.compute_llm_performance(configs, benchmarker.config_validator)
    if show:
        print_model_performance(model, *performance_metrics)


@app.command(rich_help_panel=command_categories[1])
def run_all(
        out_file: Annotated[str, typer.Argument(help='Name of file to store output (include .pkl extension)')]
):
    """
    Run all the prompts on all the LLMs and store the new data
    """
    new_df = generator.get_all_results(api_keys=keys)
    new_df.to_pickle(out_file)


if __name__ == '__main__':
    app()
