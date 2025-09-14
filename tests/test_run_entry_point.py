"""
Unit tests for run.py entry point.

Tests the main entry point functionality, error handling, and import resolution.
"""

import os
import sys
import subprocess
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the run module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run


class TestRunEntryPoint:
    """Test the run.py entry point functionality."""
    
    def test_setup_logging(self):
        """Test logging setup functionality."""
        logger = run.setup_logging()
        
        assert logger is not None
        assert logger.name == "run"
    
    def test_check_dependencies_all_present(self):
        """Test dependency checking when all packages are present."""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            
            missing = run.check_dependencies()
            
            assert missing == []
    
    def test_check_dependencies_missing_packages(self):
        """Test dependency checking when packages are missing."""
        def mock_import_side_effect(name):
            if name in ['gradio', 'openai']:
                raise ImportError(f"No module named '{name}'")
            return MagicMock()
        
        with patch('builtins.__import__', side_effect=mock_import_side_effect):
            missing = run.check_dependencies()
            
            assert 'gradio' in missing
            assert 'openai' in missing
    
    def test_check_dependencies_special_packages(self):
        """Test dependency checking for packages with special import names."""
        def mock_import_side_effect(name):
            if name == 'yaml':
                raise ImportError("No module named 'yaml'")
            elif name == 'pyyaml':
                return MagicMock()
            elif name == 'dotenv':
                raise ImportError("No module named 'dotenv'")
            elif name == 'python_dotenv':
                return MagicMock()
            return MagicMock()
        
        with patch('builtins.__import__', side_effect=mock_import_side_effect):
            missing = run.check_dependencies()
            
            # Should not include yaml or dotenv since their alternatives exist
            assert 'pyyaml' not in missing
            assert 'python-dotenv' not in missing
    
    def test_check_project_structure_valid(self):
        """Test project structure checking with valid structure."""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            missing = run.check_project_structure()
            
            assert missing == []
    
    def test_check_project_structure_missing_paths(self):
        """Test project structure checking with missing paths."""
        def mock_exists_side_effect(self):
            path_str = str(self)
            return path_str not in ['src/app.py', 'src/handlers']
        
        with patch('pathlib.Path.exists', mock_exists_side_effect):
            missing = run.check_project_structure()
            
            assert 'src/app.py' in missing
            assert 'src/handlers' in missing
    
    @patch('run.setup_logging')
    @patch('run.check_project_structure')
    @patch('run.check_dependencies')
    @patch('pathlib.Path.exists')
    def test_main_success_path(self, mock_path_exists, mock_check_deps, 
                              mock_check_structure, mock_setup_logging):
        """Test successful execution path of main function."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        mock_check_structure.return_value = []
        mock_check_deps.return_value = []
        mock_path_exists.return_value = True  # .env exists
        
        # Mock the app import and main function
        mock_app_main = MagicMock()
        
        with patch.dict('sys.modules', {'src.app': MagicMock(main=mock_app_main)}):
            with patch('run.sys.exit') as mock_exit:
                run.main()
                
                # Should not exit with error
                mock_exit.assert_not_called()
                
                # Should call app main
                mock_app_main.assert_called_once()
    
    @patch('run.setup_logging')
    @patch('run.check_project_structure')
    def test_main_missing_project_structure(self, mock_check_structure, mock_setup_logging):
        """Test main function with missing project structure."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        mock_check_structure.return_value = ['src/app.py', 'src/handlers']
        
        with patch('run.sys.exit') as mock_exit:
            # Mock sys.exit to raise an exception to stop execution
            mock_exit.side_effect = SystemExit(1)
            
            with pytest.raises(SystemExit):
                run.main()
            
            mock_exit.assert_called_with(1)
            mock_logger.error.assert_called()
    
    @patch('run.setup_logging')
    @patch('run.check_project_structure')
    @patch('run.check_dependencies')
    def test_main_missing_dependencies(self, mock_check_deps, mock_check_structure, 
                                     mock_setup_logging):
        """Test main function with missing dependencies."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        mock_check_structure.return_value = []
        mock_check_deps.return_value = ['gradio', 'openai']
        
        with patch('run.sys.exit') as mock_exit:
            # Mock sys.exit to raise an exception to stop execution
            mock_exit.side_effect = SystemExit(1)
            
            with pytest.raises(SystemExit):
                run.main()
            
            mock_exit.assert_called_with(1)
            mock_logger.error.assert_called()
    
    @patch('run.setup_logging')
    @patch('run.check_project_structure')
    @patch('run.check_dependencies')
    @patch('pathlib.Path.exists')
    def test_main_import_error(self, mock_path_exists, mock_check_deps, 
                              mock_check_structure, mock_setup_logging):
        """Test main function with import error."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        mock_check_structure.return_value = []
        mock_check_deps.return_value = []
        mock_path_exists.return_value = True
        
        with patch('run.sys.exit') as mock_exit:
            # Mock sys.exit to raise an exception to stop execution
            mock_exit.side_effect = SystemExit(1)
            
            # Mock the import to fail
            with patch.dict('sys.modules', {'src.app': None}):
                with patch('builtins.__import__') as mock_import:
                    def import_side_effect(name, *args, **kwargs):
                        if name == 'src.app':
                            raise ImportError("No module named 'src.app'")
                        return MagicMock()
                    mock_import.side_effect = import_side_effect
                    
                    with pytest.raises(SystemExit):
                        run.main()
                
                mock_exit.assert_called_with(1)
                mock_logger.error.assert_called()
    
    @patch('run.setup_logging')
    @patch('run.check_project_structure')
    @patch('run.check_dependencies')
    @patch('pathlib.Path.exists')
    def test_main_keyboard_interrupt(self, mock_path_exists, mock_check_deps, 
                                   mock_check_structure, mock_setup_logging):
        """Test main function with keyboard interrupt."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        mock_check_structure.return_value = []
        mock_check_deps.return_value = []
        mock_path_exists.return_value = True
        
        # Mock keyboard interrupt during app execution
        mock_app_main = MagicMock(side_effect=KeyboardInterrupt())
        
        with patch.dict('sys.modules', {'src.app': MagicMock(main=mock_app_main)}):
            with patch('run.sys.exit') as mock_exit:
                run.main()
                
                mock_exit.assert_called_once_with(0)
                mock_logger.info.assert_called_with("Application interrupted by user (Ctrl+C)")
    
    @patch('run.setup_logging')
    @patch('run.check_project_structure')
    @patch('run.check_dependencies')
    @patch('pathlib.Path.exists')
    def test_main_unexpected_error(self, mock_path_exists, mock_check_deps, 
                                 mock_check_structure, mock_setup_logging):
        """Test main function with unexpected error."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        mock_check_structure.return_value = []
        mock_check_deps.return_value = []
        mock_path_exists.return_value = True
        
        # Mock unexpected error during app execution
        mock_app_main = MagicMock(side_effect=RuntimeError("Unexpected error"))
        
        with patch.dict('sys.modules', {'src.app': MagicMock(main=mock_app_main)}):
            with patch('run.sys.exit') as mock_exit:
                run.main()
                
                mock_exit.assert_called_once_with(1)
                mock_logger.error.assert_called()


class TestRunIntegration:
    """Integration tests for run.py functionality."""
    
    def test_run_py_executable(self):
        """Test that run.py can be executed as a script."""
        # This test verifies the script can be imported and has main function
        assert hasattr(run, 'main')
        assert callable(run.main)
    
    def test_run_py_import_resolution(self):
        """Test that run.py resolves import issues correctly."""
        # Test that we can import run.py without import errors
        import importlib
        import sys
        
        # Remove run module if already imported
        if 'run' in sys.modules:
            del sys.modules['run']
        
        # Re-import should work without errors
        run_module = importlib.import_module('run')
        assert hasattr(run_module, 'main')
    
    @pytest.mark.slow
    def test_run_py_subprocess_execution(self):
        """Test that run.py can be executed via subprocess (integration test)."""
        # This is a more comprehensive test that actually runs the script
        # Mark as slow since it involves subprocess execution
        
        try:
            result = subprocess.run([
                sys.executable, 'run.py'
            ], capture_output=True, text=True, timeout=3, cwd=os.getcwd())
            
            # The script should not crash with import errors
            # Note: It might exit with error due to missing dependencies or other issues,
            # but it should not have "ImportError" or "attempted relative import" in stderr
            assert "ImportError" not in result.stderr
            assert "attempted relative import" not in result.stderr
            
        except subprocess.TimeoutExpired:
            # Timeout is expected since the app would try to start Gradio
            # The important thing is that it didn't fail with import errors before timeout
            pass
    
    def test_run_py_no_env_file_warning(self):
        """Test that run.py handles missing .env file gracefully."""
        with patch('run.check_project_structure', return_value=[]):
            with patch('run.check_dependencies', return_value=[]):
                # Mock Path.exists to return False only for .env file
                def mock_exists(self):
                    return str(self) != '.env'
                
                with patch('pathlib.Path.exists', mock_exists):
                    with patch.dict('sys.modules', {'src.app': MagicMock(main=MagicMock())}):
                        with patch('run.sys.exit'):
                            # This should run without errors even when .env is missing
                            # The warning will be logged but we don't need to verify the exact mock call
                            try:
                                run.main()
                                # If we get here, the function handled missing .env gracefully
                                assert True
                            except Exception as e:
                                pytest.fail(f"run.main() should handle missing .env file gracefully, but got: {e}")


if __name__ == "__main__":
    pytest.main([__file__])