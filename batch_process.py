"""
Script to process multiple claims from a file through the fact-checking system.
"""

import time
import os
import json
from dotenv import load_dotenv

from fact_check_system.workflow import process_claim
from example import format_verdict

# Load environment variables
load_dotenv()

def process_file(file_path, output_dir=None):
    """
    Process multiple claims from a file.
    
    Args:
        file_path: Path to text file with one claim per line
        output_dir: Directory to save results (defaults to 'results')
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
    
    # Set up output directory
    if output_dir is None:
        output_dir = "results"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read claims from file
    with open(file_path, 'r', encoding='utf-8') as f:
        claims = [line.strip() for line in f if line.strip()]
    
    if not claims:
        print("No claims found in the file.")
        return
    
    print(f"\nProcessing {len(claims)} claims from {file_path}...")
    
    results = []
    
    # Process each claim
    for i, claim in enumerate(claims):
        print(f"\nProcessing claim {i+1}/{len(claims)}: {claim[:50]}...")
        
        start_time = time.time()
        verdict = process_claim(claim)
        processing_time = time.time() - start_time
        
        # Add processing time to verdict
        verdict["processing_time"] = processing_time
        
        # Generate a readable report
        report = format_verdict(verdict)
        results.append({
            "claim": claim,
            "verdict": verdict,
            "report": report,
            "processing_time": processing_time
        })
        
        print(f"Completed in {processing_time:.2f} seconds")
    
    # Save results to files
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Save detailed JSON results
    json_path = os.path.join(output_dir, f"{base_filename}_{timestamp}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save readable reports
    report_path = os.path.join(output_dir, f"{base_filename}_{timestamp}_reports.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results):
            f.write(f"CLAIM {i+1}: {result['claim']}\n")
            f.write("=" * 80 + "\n")
            f.write(result['report'])
            f.write("\n\n" + "=" * 80 + "\n\n")
    
    print(f"\nProcessing complete. Results saved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - Reports: {report_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process multiple claims from a file.")
    parser.add_argument("file_path", help="Path to text file with one claim per line")
    parser.add_argument("--output", "-o", help="Directory to save results (defaults to 'results')")
    
    args = parser.parse_args()
    
    process_file(args.file_path, args.output) 