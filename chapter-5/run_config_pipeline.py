#!/usr/bin/env python3
"""
Simple verification script demonstrating Chapter 17's configuration workflow.
"""

from modules.config import (
    get_default_config,
    merge_configs,
    validate_config,
    apply_environment_profile
)

def main():
    print("=" * 60)
    print("RUNNING CHAPTER 17 CONFIGURATION SYSTEM DEMO")
    print("=" * 60)

    # 1. Fetch system baseline defaults
    config = get_default_config()
    print("[1] Initialized system with default BASE_CONFIG keys.")

    # 2. Inject experiment-specific overrides
    overrides = {
        "experiment_id": "demo_experiment_v1",
        "data": {
            "url": "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv",
            "target": "Sales"
        },
        "splitting": {
            "test_size": 0.25  # Changing test split from default 0.20 to 0.25
        },
        "models": {
            "linear_regression": {
                "class": "LinearRegression",
                "params": {}
            }
        }
    }
    
    config = merge_configs(config, overrides)
    print("[2] Successfully merged experiment overrides into base config.")
    print(f"    - Target Split Size: {config['splitting']['test_size']}")

    # 3. Assert structural integrity via schema validation
    errors = validate_config(config)
    if errors:
        print("[!] Validation failed with the following errors:")
        for err in errors:
            print(f"    - {err}")
        return
    else:
        print("[3] Config schema validation passed successfully (0 errors found).")

    # 4. Resolve runtime environment profile overlays
    # Simulates environment routing (e.g., swapping directories for testing or production)
    final_config = apply_environment_profile(config, environment="testing")
    print("[4] Applied environment profile overlay successfully.")
    print(f"    - Registry target path redirected to: {final_config['output']['registry_path']}")
    print(f"    - Active Runtime Environment flag:   {final_config['_environment']}")

    print("=" * 60)
    print("DEMO EXECUTED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()