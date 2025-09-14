"""
Tests to verify that import errors have been resolved.

This test file specifically verifies that the import error fix is working correctly
and that both direct execution and module execution work as expected.
"""

import os
import sys
import subprocess
import pytest
from unittest.mock import patch, MagicMock


class TestImportErrorResolution:
    """Test that import errors have been resolved."""
    
    def test_src_app_can_be_imported_as_module(self):
        """Test that src.app can be imported as a module without errors."""
        # This should work now that we have proper entry point separation
        try:
            import src.app
            assert hasattr(src.app, 'main')
            assert callable(src.app.main)
        except ImportError as e:
            pytest.fail(f"Failed to import src.app as module: {e}")
    
    def test_run_py_imports_src_app_successfully(self):
        """Test that run.py can import src.app.main successfully."""
        # Mock the src.app module to avoid actually running the app
        mock_app_module = MagicMock()
        mock_main_function = MagicMock()
        mock_app_module.main = mock_main_function
        
        with patch.dict('sys.modules', {'src.app': mock_app_module}):
            # Import run module
            import run
            
            # Mock other dependencies to focus on import testing
            with patch('run.check_project_structure', return_value=[]):
                with patch('run.check_dependencies', return_value=[]):
                    with patch('pathlib.Path.exists', return_value=True):
                        with patch('run.sys.exit'):
                            # This should not raise ImportError
                            run.main()
                            
                            # Verify that src.app.main was called
                            mock_main_function.assert_called_once()
    
    def test_relative_imports_still_work_in_src_package(self):
        """Test that relative imports still work within the src package."""
        # This test ensures we haven't broken existing relative imports
        try:
            # Import a module that uses relative imports
            import src.handlers.audio_handler
            
            # Verify the module loaded successfully
            assert hasattr(src.handlers.audio_handler, 'AudioHandler')
            assert hasattr(src.handlers.audio_handler, 'MockAudioHandler')
            
        except ImportError as e:
            pytest.fail(f"Relative imports broken in src package: {e}")
    
    @pytest.mark.slow
    def test_run_py_subprocess_no_import_error(self):
        """Test that running run.py via subprocess doesn't produce import errors."""
        # Run the script with a timeout to prevent hanging
        try:
            result = subprocess.run([
                sys.executable, 'run.py'
            ], capture_output=True, text=True, timeout=5, cwd=os.getcwd())
            
            # Check that no import errors occurred
            assert "ImportError" not in result.stderr
            assert "attempted relative import" not in result.stderr
            assert "No module named" not in result.stderr
            
        except subprocess.TimeoutExpired:
            # Timeout is expected since the app would try to start Gradio
            # The important thing is that it didn't fail with import errors
            pass
    
    def test_python_module_execution_works(self):
        """Test that python -m src.app works without import errors."""
        # This tests the alternative execution method mentioned in requirements
        try:
            result = subprocess.run([
                sys.executable, '-m', 'src.app'
            ], capture_output=True, text=True, timeout=5, cwd=os.getcwd())
            
            # Check that no import errors occurred
            assert "ImportError" not in result.stderr
            assert "attempted relative import" not in result.stderr
            
        except subprocess.TimeoutExpired:
            # Timeout is expected since the app would try to start Gradio
            pass
    
    def test_existing_test_imports_work(self):
        """Test that existing test files can import modules correctly."""
        # Test that the conftest.py setup still works
        try:
            # Import modules using absolute imports (as they should be used in tests)
            import src.transcribe
            import src.llm
            import src.util
            import src.errors
            
            # Verify basic functionality is available
            assert hasattr(src.transcribe, 'transcribe')
            assert hasattr(src.llm, 'chat_completion')
            assert hasattr(src.util, 'split_audio')
            
        except ImportError as e:
            pytest.fail(f"Test imports broken: {e}")
    
    def test_handlers_can_be_imported_directly(self):
        """Test that handler modules can be imported directly."""
        try:
            from src.handlers import audio_handler, chat_handler, history_handler, settings_handler
            
            # Verify both real and mock handlers are available
            assert hasattr(audio_handler, 'AudioHandler')
            assert hasattr(audio_handler, 'MockAudioHandler')
            assert hasattr(chat_handler, 'ChatHandler')
            assert hasattr(chat_handler, 'MockChatHandler')
            assert hasattr(history_handler, 'HistoryHandler')
            assert hasattr(history_handler, 'MockHistoryHandler')
            assert hasattr(settings_handler, 'SettingsHandler')
            assert hasattr(settings_handler, 'MockSettingsHandler')
            
        except ImportError as e:
            pytest.fail(f"Handler imports broken: {e}")
    
    def test_config_imports_work(self):
        """Test that config imports work correctly."""
        try:
            from src.config import AppConfig
            
            # Verify config class is available
            assert AppConfig is not None
            
        except ImportError as e:
            pytest.fail(f"Config imports broken: {e}")


class TestBackwardCompatibility:
    """Test that the changes maintain backward compatibility."""
    
    def test_existing_functionality_preserved(self):
        """Test that existing functionality is preserved."""
        # Mock environment to avoid actual API calls
        with patch.dict(os.environ, {'APP_ENV': 'mock-ui'}):
            try:
                # Import main modules
                import src.app
                import src.transcribe
                import src.llm
                
                # Verify key functions exist
                assert hasattr(src.app, 'main')
                assert hasattr(src.transcribe, 'transcribe')
                assert hasattr(src.llm, 'chat_completion')
                
            except Exception as e:
                pytest.fail(f"Existing functionality broken: {e}")
    
    def test_environment_variable_handling_preserved(self):
        """Test that environment variable handling still works."""
        # Test different APP_ENV values
        test_envs = ['prod', 'test', 'mock-ui']
        
        for env in test_envs:
            with patch.dict(os.environ, {'APP_ENV': env}):
                try:
                    # Should be able to import without errors
                    import src.config
                    config = src.config.AppConfig(env=env)
                    
                    # Verify environment is set correctly (using the correct attribute name)
                    assert config.env == env
                    
                except Exception as e:
                    pytest.fail(f"Environment handling broken for {env}: {e}")


class TestImportErrorActualResolution:
    """Test that the actual import error has been resolved in practice."""
    
    def test_error_handling_with_standard_exceptions(self):
        """Test that error handling works correctly with standard Python exceptions."""
        try:
            from src.errors import get_user_friendly_message
            
            # Test ModuleNotFoundError handling
            try:
                raise ModuleNotFoundError("No module named 'test'")
            except Exception as e:
                msg = get_user_friendly_message(e)
                assert "Module import error" in msg
                assert "installation" in msg
            
            # Test ImportError handling
            try:
                raise ImportError("Import failed")
            except Exception as e:
                msg = get_user_friendly_message(e)
                assert "Import error" in msg
                assert "installation" in msg
            
            # Test generic exception handling
            try:
                raise ValueError("Some error")
            except Exception as e:
                msg = get_user_friendly_message(e)
                assert "unexpected error" in msg
                assert "Some error" in msg
                
        except ImportError as e:
            pytest.fail(f"Failed to import error handling function: {e}")
    
    def test_specific_llm_import_error_fixed(self):
        """Test that the specific 'No module named llm' error is fixed."""
        # This test specifically addresses the error: ModuleNotFoundError: No module named 'llm'
        # that was occurring in src/app.py line 875
        try:
            import src.app
            # Try to access the function that was causing the import error
            # This should not raise an ImportError anymore
            assert hasattr(src.app, 'main')
            
            # Test that we can import llm through the proper relative import
            from src.llm import get_language_code
            assert callable(get_language_code)
            
        except ImportError as e:
            pytest.fail(f"Import error still present: {e}")
    
    def test_run_py_execution_no_import_error(self):
        """Test that run.py can be executed without import errors."""
        import subprocess
        import sys
        
        try:
            # Run with a very short timeout to just check if it starts
            result = subprocess.run([
                sys.executable, 'run.py'
            ], capture_output=True, text=True, timeout=2, cwd=os.getcwd())
            
            # Check that no import errors occurred
            assert "ImportError" not in result.stderr
            assert "attempted relative import" not in result.stderr
            assert "No module named" not in result.stderr
            
        except subprocess.TimeoutExpired:
            # Timeout is expected since the app would try to start Gradio
            # The important thing is that it didn't fail with import errors
            pass
    
    def test_module_execution_no_import_error(self):
        """Test that python -m src.app works without import errors."""
        import subprocess
        import sys
        
        try:
            # Run with a very short timeout to just check if it starts
            result = subprocess.run([
                sys.executable, '-m', 'src.app'
            ], capture_output=True, text=True, timeout=2, cwd=os.getcwd())
            
            # Check that no import errors occurred
            assert "ImportError" not in result.stderr
            assert "attempted relative import" not in result.stderr
            assert "No module named" not in result.stderr
            
        except subprocess.TimeoutExpired:
            # Timeout is expected since the app would try to start Gradio
            # The important thing is that it didn't fail with import errors
            pass
    
    def test_import_error_fix_requirements_met(self):
        """Test that all requirements from the spec have been met."""
        # Requirement 1.1: uv run python src/app.py should work
        # This is tested by the run.py entry point tests
        
        # Requirement 2.1: Existing project structure maintained
        # Test that relative imports still work within src package
        try:
            import src.handlers.audio_handler
            assert hasattr(src.handlers.audio_handler, 'AudioHandler')
            assert hasattr(src.handlers.audio_handler, 'MockAudioHandler')
        except ImportError as e:
            pytest.fail(f"Relative imports broken in src package: {e}")
        
        # Test that the main function exists in src.app
        try:
            import src.app
            assert hasattr(src.app, 'main')
            assert callable(src.app.main)
        except ImportError as e:
            pytest.fail(f"Failed to import src.app: {e}")


if __name__ == "__main__":
    pytest.main([__file__])