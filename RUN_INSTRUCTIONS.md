# Real-Time Voice Cloning - Setup and Run Instructions

## 1. Set up Python Virtual Environment (Recommended)
Open a terminal in the project directory `c:/Users/lab302/Desktop/Real-Time-Voice-Cloning-master` and run:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows (cmd):
  ```
  venv\Scripts\activate
  ```
- On Windows (PowerShell):
  ```
  .\venv\Scripts\Activate.ps1
  ```

## 2. Install Dependencies

Make sure you have Python 3.7+ installed.

Install PyTorch following instructions at https://pytorch.org/get-started/locally/ depending on your system and whether you have a GPU or not. For example, for CPU only:

```bash
pip install torch torchvision torchaudio
```

Install ffmpeg:

- Download and install ffmpeg from https://ffmpeg.org/download.html#get-packages
- Make sure ffmpeg is added to your system PATH so it can be accessed from the command line.

Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

## 3. Test Configuration with demo_cli.py

Run the following command to test the setup and interactively generate speech:

```bash
python demo_cli.py
```

Follow the prompts to provide a reference audio file and text to synthesize.

## 4. Launch the Toolbox GUI with demo_toolbox.py

To launch the GUI toolbox, run:

```bash
python demo_toolbox.py
```

Optionally, if you have datasets downloaded, specify the datasets root directory:

```bash
python demo_toolbox.py -d <datasets_root>
```

## Notes

- Pretrained models will be downloaded automatically if not present in the `saved_models` directory.
- If you want to run on CPU only, add the `--cpu` flag to the commands.
- For any issues related to audio playback, you can use the `--no_sound` flag with `demo_cli.py`.

This should get you started with running the Real-Time Voice Cloning project.
