# Emotion-Guided VLM Analysis System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[username]/BADNet_Demo/blob/main/emotion_guided_vlm/emotion_guided_vlm_demo_v2.ipynb)

**One-Click Demo: Click the badge above to open in Google Colab and start immediately!**

Advanced multi-person emotion detection with AI-powered scene analysis - no installation required!

## Features

- **Multi-person emotion detection** with enhanced validation filters
- **Group emotion aggregation** using multiple methods (majority, weighted, dominant)
- **Emotion-guided VLM prompting** for context-aware scene analysis using BLIP
- **Interactive face selection** with manual coordinate entry and visual feedback
- **4-panel visualization** with emotion distribution and analysis comparison
- **Live webcam analysis** with real-time processing
- **Cross-platform compatibility** optimized for Google Colab

## Getting Started (Zero Installation!)

### Method 1: Direct Launch (Recommended)
1. **Click the Colab badge above** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
2. **Allow camera permissions** when prompted (for webcam features)
3. **Run all cells** using `Runtime → Run All`
4. **Start analyzing!** Use the interactive menu that appears

### Method 2: Manual Setup
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload the `emotion_guided_vlm_demo_v2.ipynb` file
3. Run all cells and enjoy!

### What You Need
- ✓ Google account (for Colab access)
- ✓ Web browser with camera permissions (for webcam features)
- ✓ Images to upload (optional - we provide examples)
- ✗ No installation required!
- ✗ No Python environment setup needed!

## How to Use

### Step-by-Step Workflow

#### 1. Initialize the System (First cell)
```python
# Cell 1: Run this first - loads all models automatically
pipeline = initialize_models()
```
*Takes 30-60 seconds on first run to download AI models*

#### 2. Choose Your Analysis Mode
```python
# Cell 2: Launch interactive menu
main_interactive_demo()
```

#### 3. Select Demo Type
The interactive menu offers:
- **Quick Auto Analysis** - Upload images for automatic detection
- **Manual Face Selection** - Click and drag to select faces precisely  
- **Live Webcam Demo** - Real-time emotion analysis
- **Test Tools** - Practice with the interface

### Pro Tips for Colab
- **Runtime Management**: Choose "GPU" runtime for faster processing
- **File Uploads**: Images under 25MB work best
- **Browser**: Chrome/Firefox work best for webcam features
- **Permissions**: Allow camera access for live demo features

## Analysis Capabilities

### Emotion Detection
- **7 emotion categories**: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- **Multi-person support**: Analyze groups with individual + collective results  
- **Enhanced validation**: Filters false positives with contrast/aspect ratio checks
- **Confidence thresholding**: Configurable minimum confidence levels

### VLM Scene Analysis  
- **Standard captioning**: General scene description using BLIP
- **Emotion-guided prompts**: Context-aware analysis based on detected emotions
- **Workplace scenarios**: Specialized prompts for productivity/safety analysis
- **Comparative results**: Side-by-side standard vs. emotion-guided analysis

### Visualization System
- **4-panel display**: Annotated image, emotion distribution, VLM comparison, pipeline flow
- **Interactive selection**: Mouse-based region selection with visual feedback
- **Coordinate validation**: Grid overlay and bounds checking for manual entry
- **Clean results display**: Professional formatting with clear indicators

## Demo Modes Comparison

| Mode | Environment | Input | Face Selection | Best For |
|------|-------------|-------|----------------|----------|
| Quick Auto | Colab | Upload | Automatic | Fast analysis, good lighting |
| Interactive Manual | Colab | Upload | Mouse drag | Precise control, complex scenes |
| Advanced Interactive | Local/Colab | File path | Manual coordinates | Professional use, batch processing |
| Live Webcam | Colab | Camera | Automatic | Real-time demos, presentations |
| Test Selection | Colab | Upload | Practice only | Learning the interface |

### When to Use Each Mode
- **Quick Auto**: Good lighting, clear faces, simple scenes
- **Manual Selection**: Poor lighting, partial faces, crowded scenes  
- **Advanced**: Batch processing, precise analysis, research use
- **Webcam**: Live demos, real-time monitoring, interactive presentations

## Notebook Structure

The notebook is organized in logical sections:

### Setup & Initialization (Cells 1-2)
- Import dependencies and load AI models
- One-time setup (takes ~60 seconds first run)

### Core Analysis Engine (Cells 3-4)
- Multi-person emotion detection system
- VLM scene analysis with emotion guidance

### Interactive Features (Cells 5-6)
- Manual face selection tools
- Upload and processing workflows

### Live Demo (Cell 7)
- Real-time webcam analysis
- Interactive capture and analysis

### Running Tips
- **First time**: Run `Runtime → Run All` and wait for models to load
- **Regular use**: Just run `main_interactive_demo()` after initialization
- **Troubleshooting**: Restart runtime and run setup cells again

## Configuration Options

```python
# Customize detection parameters
CONFIG = {
    'scale_factor': 1.05,              # Face detection sensitivity
    'min_neighbors': 8,                # False positive filtering
    'min_face_size': (40, 40),         # Minimum detectable face
    'max_face_size': (300, 300),       # Maximum detectable face
    'emotion_confidence_threshold': 30.0, # Minimum emotion confidence
    'aggregation_method': 'weighted_average', # Group emotion method
    'distance_weight': True,           # Weight by face size
    'max_vlm_length': 50,              # VLM response length
    'device': 'cuda'                   # GPU acceleration if available
}
```

## Troubleshooting

### Colab-Specific Issues

#### "Runtime Disconnected"
```python
# Reconnect and run this cell to restore your session
pipeline = initialize_models()
main_interactive_demo()
```

#### Webcam Not Working
- Allow camera permissions in browser
- Use Chrome or Firefox (Safari can be problematic)
- Try refreshing the page and rerunning cells
- Check browser console for JavaScript errors

#### File Upload Issues  
- Image size < 25MB (Colab limit)
- Supported formats: JPG, PNG, JPEG
- Clear browser cache if uploads fail
- Try incognito/private mode

#### Model Loading Errors
```python
# If models fail to load, try this reset:
import torch
torch.cuda.empty_cache()  # Clear GPU memory
pipeline = initialize_models()  # Reload models
```

#### Slow Performance
- Switch to GPU runtime: `Runtime → Change runtime type → GPU`
- Close other Colab notebooks to free memory
- Restart runtime: `Runtime → Restart runtime`
- Use smaller images (resize before upload)

### Common Usage Issues

#### "No faces detected"
- Ensure good lighting on faces
- Try manual selection mode for difficult images
- Check face size (minimum 40×40 pixels)
- Upload different image for comparison

#### Manual selection not working
- Click and drag to create rectangles
- Press ENTER when done selecting
- Use ESC to remove last selection
- Ensure coordinates are within image bounds

## Example Usage

### Process Single Image
```python
# Initialize once
pipeline = initialize_models()

# Analyze image
results = pipeline.process_image('path/to/image.jpg')
pipeline.display_clean_results(results)
```

### Custom Face Regions
```python
# Manual face coordinates (x, y, width, height)
custom_faces = [(100, 50, 150, 150), (300, 75, 120, 120)]
results = pipeline.process_image('image.jpg', custom_faces=custom_faces)
```

### Group Emotion Analysis
```python
# Different aggregation methods
CONFIG['aggregation_method'] = 'weighted_average'  # 'majority', 'dominant'
results = pipeline.process_image('group_photo.jpg')
```

## Ready to Try It?

### Start Now: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[username]/BADNet_Demo/blob/main/emotion_guided_vlm/emotion_guided_vlm_demo_v2.ipynb)

### Quick Checklist
- [ ] Click the Colab badge above
- [ ] Run all cells (Runtime → Run All)  
- [ ] Wait for model initialization (~60 seconds)
- [ ] Launch `main_interactive_demo()`
- [ ] Choose your analysis mode and start!

### Demo Modes to Try
1. **Start simple**: Quick Auto Analysis with a clear portrait photo
2. **Get advanced**: Manual selection with a group photo
3. **Go live**: Webcam demo for real-time analysis
4. **Experiment**: Try different lighting conditions and scenarios

## Technical Details

### Dependencies
- PyTorch and Transformers (for BLIP VLM)
- OpenCV (for face detection and image processing)
- DeepFace (for emotion analysis)
- Matplotlib (for interactive visualization)
- NumPy, PIL (for image manipulation)

### AI Models Used
- **Face Detection**: OpenCV Haar Cascades with validation filters
- **Emotion Analysis**: DeepFace with multiple backend support
- **VLM Analysis**: Salesforce BLIP image captioning model
- **Processing**: Custom pipeline with group aggregation algorithms

### Performance Notes
- First run downloads ~500MB of AI models
- GPU runtime recommended for faster processing
- Memory usage: ~2-4GB RAM typical
- Processing time: 2-5 seconds per image analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details. 