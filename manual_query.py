"""
Script to run manual queries through the fact-checking system.
"""

import time
from dotenv import load_dotenv

from fact_check_system.workflow import process_claim
from example import format_verdict

# Load environment variables
load_dotenv()

def run_manual_query():
    """Process a manually entered claim."""
    print("\n===== Tathya Fact-Checking System =====\n")
    print("Enter a claim to fact-check (or 'exit' to quit):\n")
    
    while True:
        # Get user input
        claim = input("> ")
        
        # Check if user wants to exit
        if claim.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the Tathya Fact-Checking System.")
            break
        
        # Skip empty inputs
        if not claim.strip():
            continue
        
        print("\nProcessing claim, please wait...\n")
        
        # Process the claim
        start_time = time.time()
        verdict = process_claim(claim)
        processing_time = time.time() - start_time
        
        # Display results
        print("\n" + "=" * 80)
        print(format_verdict(verdict))
        print("-" * 80)
        print(f"Processing time: {processing_time:.2f} seconds")
        print("=" * 80)
        
        print("\nEnter another claim to fact-check (or 'exit' to quit):\n")

if __name__ == "__main__":
    run_manual_query() 