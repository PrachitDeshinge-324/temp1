"""
Command-line interface for gait recognition system
"""

from gait_config import parse_args
import gait_main

def main():
    """Main entry point for CLI"""
    # Parse arguments
    config = parse_args()
    
    # Create output directory
    import os
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Run main function
    gait_main.run(config)

if __name__ == "__main__":
    main()
