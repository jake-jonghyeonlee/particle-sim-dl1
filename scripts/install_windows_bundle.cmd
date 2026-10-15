@echo off
setlocal DisableDelayedExpansion
rem Run in CMD. Python is preinstalled; this script never downloads or elevates.
py -3.12 -I "%~dp0install_windows_bundle.py" %*
exit /b %errorlevel%
