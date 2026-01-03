# LLM-Driven Database Connector Config Generator & Benchmarking Tool

## Overview

This project is a CLI-based tool designed to leverage Large Language Models (LLMs) to generate standardized, application-consumable database connector configurations. It addresses the challenge of inconsistent configuration formats across different database technologies.

In addition to generation, the tool includes a robust **benchmarking engine** that evaluates the performance of different LLMs (ChatGPT, Claude, Gemini, Mistral) and various prompt engineering strategies. It scores outputs based on strict JSON schema validation, completeness, and formatting correctness.

## Key Features

* **Multi-LLM Support:** Integrates with OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), Google (Gemini 1.5 Flash), and Mistral (Large).
* **Standardized Schema:** Enforces a strict JSON schema for 7 major databases: PostgreSQL, MySQL, SQL Server, MongoDB, Snowflake, BigQuery, and Oracle.
* **Prompt Benchmarking:** Evaluates prompts based on a validity score (0-100), penalizing schema violations and error depth.
* **Model Benchmarking:** Calculates Precision, Recall, and F1-Scores for each LLM to measure reliability and accuracy.
* **Rich CLI:** Built with `Typer` and `Rich` for beautiful, interactive terminal output and data visualization.
