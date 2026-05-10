"""
Wrapper script to test classification setup from root directory
"""

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import and run the test
    from classification.test_classification_setup import main
    main()
