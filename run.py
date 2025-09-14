#!/usr/bin/env python3
"""Entry point for the transcriber web app."""

import logging
import sys
import os
from pathlib import Path

def setup_logging():
    """Setup basic logging configuration for error reporting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'gradio',
        'openai', 
        'pydub',
        'pydantic',
        'yaml',
        'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            # Handle special cases for package names
            if package == 'yaml':
                try:
                    __import__('pyyaml')
                except ImportError:
                    missing_packages.append('pyyaml')
            elif package == 'dotenv':
                try:
                    __import__('python_dotenv')
                except ImportError:
                    missing_packages.append('python-dotenv')
            else:
                missing_packages.append(package)
    
    return missing_packages

def check_project_structure():
    """Verify that the project structure is correct."""
    required_paths = [
        'src',
        'src/app.py',
        'src/config',
        'src/handlers'
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    return missing_paths

def main():
    """Main entry point with comprehensive error handling."""
    logger = setup_logging()
    
    try:
        logger.info("Starting transcriber web app...")
        
        # Check project structure
        logger.info("Checking project structure...")
        missing_paths = check_project_structure()
        if missing_paths:
            logger.error("Missing required project files/directories:")
            for path in missing_paths:
                logger.error(f"  - {path}")
            logger.error("Please ensure you're running from the project root directory.")
            sys.exit(1)
        
        # Check dependencies
        logger.info("Checking dependencies...")
        missing_packages = check_dependencies()
        if missing_packages:
            logger.error("Missing required dependencies:")
            for package in missing_packages:
                logger.error(f"  - {package}")
            logger.error("Please install dependencies with: pip install -r requirements.txt")
            sys.exit(1)
        
        # Check environment configuration
        logger.info("Checking environment configuration...")
        env_file = Path('.env')
        if not env_file.exists():
            logger.warning("No .env file found. Using default configuration.")
            logger.info("For production use, copy sample.env to .env and configure your settings.")
        
        # Import and run the main application
        logger.info("Importing application modules...")
        try:
            from src.app import main as app_main
        except ImportError as e:
            logger.error(f"Failed to import main function from src.app: {e}")
            logger.error("This usually indicates:")
            logger.error("  1. Missing dependencies (check above)")
            logger.error("  2. Syntax errors in src/app.py or its imports")
            logger.error("  3. Missing main() function in src/app.py")
            logger.error("Please check the application code for errors.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error during import: {e}")
            logger.error("This may indicate a configuration or environment issue.")
            sys.exit(1)
        
        # Run the application
        logger.info("Starting Gradio application...")
        app_main()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error("Please check the logs above for more details.")
        logger.error("If the problem persists, please check:")
        logger.error("  1. Your .env configuration")
        logger.error("  2. Network connectivity (for API calls)")
        logger.error("  3. File permissions in the project directory")
        sys.exit(1)

if __name__ == "__main__":
    main()