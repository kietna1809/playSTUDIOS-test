#!/usr/bin/env python3
"""
Test script for fraud detection API server.
Reads test.csv and sends each row to the /detect endpoint.

Usage:
    python tests/test_api.py --host localhost --port 8000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests


def load_test_data(csv_path: str) -> List[Dict[str, Any]]:
    """Load test data from CSV file using pandas."""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Define feature columns (all except Class)
    feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Extract features and labels
    test_cases = []
    for idx, row in df.iterrows():
        features = row[feature_columns].values.tolist()
        expected_class = int(row['Class'])
        
        test_cases.append({
            'features': [features],
            'expected_class': expected_class
        })
    return test_cases


def test_api(host: str, port: int, test_data: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, Any]:
    """Test the fraud detection API with test data."""
    url = f"http://{host}:{port}/detect"
    
    total = len(test_data)
    successful = 0
    failed = 0
    correct_predictions = 0
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    errors = []
    
    print(f"Testing API at {url}")
    print(f"Total test cases: {total}")
    print("-" * 80)
    
    for idx, test_case in enumerate(test_data, 1):
        payload = {"features": test_case['features']}
        expected_class = test_case['expected_class']
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                successful += 1
                result = response.json()
                predicted_class = result['result'][0]
                
                # Check if prediction matches expected
                if predicted_class == expected_class:
                    correct_predictions += 1
                
                # Calculate confusion matrix values
                if predicted_class == 1 and expected_class == 1:
                    true_positives += 1
                elif predicted_class == 0 and expected_class == 0:
                    true_negatives += 1
                elif predicted_class == 1 and expected_class == 0:
                    false_positives += 1
                elif predicted_class == 0 and expected_class == 1:
                    false_negatives += 1
                
                if verbose:
                    status = "✓" if predicted_class == expected_class else "✗"
                    print(f"[{idx}/{total}] {status} Expected: {expected_class}, Predicted: {predicted_class}")
                elif idx % 10 == 0:
                    print(f"Processed {idx}/{total} samples...")
                    
            else:
                failed += 1
                errors.append(f"Test {idx}: HTTP {response.status_code} - {response.text}")
                if verbose:
                    print(f"[{idx}/{total}] ✗ HTTP Error {response.status_code}")
                    
        except requests.exceptions.RequestException as e:
            failed += 1
            errors.append(f"Test {idx}: {str(e)}")
            if verbose:
                print(f"[{idx}/{total}] ✗ Request failed: {e}")
    
    # Calculate metrics
    accuracy = (correct_predictions / total * 100) if total > 0 else 0
    precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    results = {
        'total': total,
        'successful': successful,
        'failed': failed,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'errors': errors
    }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print test results in a formatted way."""
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests:        {results['total']}")
    print(f"Successful:         {results['successful']}")
    print(f"Failed:             {results['failed']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Accuracy:           {results['accuracy']:.2f}%")
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print(f"True Positives:     {results['true_positives']}")
    print(f"True Negatives:     {results['true_negatives']}")
    print(f"False Positives:    {results['false_positives']}")
    print(f"False Negatives:    {results['false_negatives']}")
    print("\n" + "-" * 80)
    print("METRICS")
    print("-" * 80)
    print(f"Precision:          {results['precision']:.2f}%")
    print(f"Recall:             {results['recall']:.2f}%")
    print(f"F1 Score:           {results['f1_score']:.2f}%")
    
    if results['errors']:
        print("\n" + "-" * 80)
        print("ERRORS")
        print("-" * 80)
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"  • {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test the fraud detection API server with test data."
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "localhost"),
        help="API server host (default: localhost or HOST env var)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "6060")),
        help="API server port (default: 6060 or PORT env var)"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to test CSV file (default: tests/test.csv)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output showing each test result"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Determine CSV path
    if args.csv:
        csv_path = args.csv
    else:
        # Try to find test.csv relative to script location
        script_dir = Path(__file__).parent
        csv_path = script_dir / "test.csv"
        
        if not csv_path.exists():
            # Try relative to current working directory
            csv_path = Path("tests/test.csv")
    
    # Check if CSV exists
    if not Path(csv_path).exists():
        print(f"Error: Test CSV file not found at {csv_path}", file=sys.stderr)
        print("Please provide the path using --csv argument.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading test data from: {csv_path}")
    
    try:
        test_data = load_test_data(str(csv_path))
        print(f"Loaded {len(test_data)} test cases\n")
    except Exception as e:
        print(f"Error loading test data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run tests
    try:
        results = test_api(args.host, args.port, test_data, args.verbose)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    
    # Print results
    print_results(results)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with error code if tests failed
    if results['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

