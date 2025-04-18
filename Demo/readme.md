# BADNet Demo Tools

This repository contains Python modules for real-time visual analysis:

1. **Face Emotion Analyzer** - Detects faces and recognizes emotions using DeepFace
2. **Vision Language Model Demo** - Performs general scene analysis using Ollama and vision-language models

## Requirements

- Python 3.8+
- Miniconda or similar environment manager (recommended)
- Webcam
- For VLM Demo: Ollama installed locally

## Installation

### 1. Set up environment

```bash
# Create a new conda environment
conda create -n badnet_demo python=3.11
conda activate badnet_demo

# Install dependencies
pip install -r requirements.txt
```

### 2. For VLM Demo: Install Ollama

#### macOS
```bash
brew install ollama
```

#### Windows
Download from [Ollama website](https://ollama.com/download/windows)

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. For VLM Demo: Download a vision model

Open a terminal and run:
```bash
# Start the Ollama service
ollama serve

# In a different terminal, download a vision model
ollama pull llava:latest
```

## Usage

### Face Emotion Analyzer

Run the facial emotion detection module:

```bash
python face_demo2.py
```

This will:
- Open your webcam
- Detect faces in the video feed
- Analyze the emotions of detected faces
- Display color-coded bounding boxes according to emotions

Press 'q' to quit.

### Vision Language Model Demo

Start the Ollama service in a terminal:
```bash
ollama serve
```

Then in a new terminal, run the VLM demo:
```bash
python vlm_demo.py --model llava:latest --fps 0.2
```

Command line options:
- `--model`: Name of the Ollama vision model (default: llama3-2-vision)
- `--prompt`: Custom instructions for scene analysis
- `--camera`: Camera device ID (default: 0)
- `--fps`: Analysis frequency in frames per second (default: 0.2 - once every 5 seconds)
- `--ollama-url`: Custom Ollama API URL if not using the default

Controls:
- Press 'A' to force immediate analysis of the current frame
- Press 'Q' to quit

## Example Prompts

You can customize the VLM analysis with different prompts:

```bash
# Detect safety hazards
python vlm_demo.py --model llava:latest --prompt "Identify any safety hazards or dangerous situations in this scene."

# Count people
python vlm_demo.py --model llava:latest --prompt "Count the number of people in the scene and describe what they are doing."

# Describe environment
python vlm_demo.py --model llava:latest --prompt "Describe the environment in detail, focusing on the layout and objects."
```

## Troubleshooting

### Camera not detected
- Verify your webcam is connected and functioning
- Try a different camera ID: `--camera 1`

### VLM not working
- Ensure Ollama is running (`ollama serve`)
- Verify you've downloaded the model (`ollama list`)
- Try with a model you know exists (`ollama pull llava:latest`)

### DeepFace errors
- If you encounter MTCNN or RetinaFace errors, try reinstalling:
  ```bash
  pip uninstall deepface
  pip install deepface
  ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
