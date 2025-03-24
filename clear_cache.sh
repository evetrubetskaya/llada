python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]" 
python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]" 
python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('.ipynb_checkpoints/*')]" 
python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('.ipynb_checkpoints')]"
