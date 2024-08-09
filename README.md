# Garden-Watch
Prerequisites:
- Python


Build instructions:
```
    pip install -r requirements.txt
    pyinstaller --onefile --windowed --icon=icon.ico --add-data "gardenwatch.wav;." "Garden Watch.py"
```