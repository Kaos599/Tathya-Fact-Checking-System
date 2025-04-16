# Tathya Fact-Checking System: Manual Queries

This guide explains how to run the Tathya Fact-Checking System with your own queries.

## Prerequisites

Ensure you have installed all required dependencies and set up API keys in the `.env` file. See the main README for setup instructions.

## Running Manual Queries

The `manual_query.py` script allows you to interactively enter claims for fact-checking.

```bash
python manual_query.py
```

This will start an interactive session where you can:
- Enter a claim to fact-check
- View the detailed verdict with sources and intermediate steps
- Continue with additional queries or exit

## Batch Processing Claims

The `batch_process.py` script processes multiple claims from a text file (one claim per line).

```bash
python batch_process.py sample_claims.txt
```

Options:
- `--output` or `-o`: Specify a custom output directory (defaults to `results`)

Example:
```bash
python batch_process.py sample_claims.txt --output my_results
```

### Output Files

Batch processing generates two output files:
1. A JSON file with detailed results for each claim
2. A text file with human-readable reports

Files are named with a timestamp to avoid overwriting previous results.

## Understanding the Output

The fact-checking system provides detailed output that includes:

1. **Overall Verdict**: The final assessment of the claim (TRUE, FALSE, PARTIALLY_TRUE, UNVERIFIABLE, NEEDS_CONTEXT)
2. **Confidence Score**: How confident the system is in its verdict (0.0-1.0)
3. **Summary**: A concise explanation of the verdict
4. **Fact-Checking Process**: Step-by-step breakdown of how the claim was evaluated
   - Claim decomposition into atomic verifiable units
   - Evidence collection from multiple sources
   - Verification by multiple specialized agents
   - Final adjudication and synthesis
5. **Sources Used**: All sources consulted during fact-checking
6. **Atomic Claims**: Individual verifiable claims extracted from the input, with their own verdicts and evidence

## Examples

Sample claims for testing:
- "The Earth orbits the Sun."
- "COVID-19 vaccines have been linked to thousands of deaths, and the mRNA technology used in the Pfizer and Moderna vaccines can alter human DNA."
- "The Great Wall of China is visible from the Moon with the naked eye."
- "Humans only use 10% of their brains." 