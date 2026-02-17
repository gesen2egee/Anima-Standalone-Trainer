# Anima Standalone Trainer

A lightweight, decoupled training environment for circlestone-labs' Anima model, currently support Lora training only. Built upon [sd-scripts](https://github.com/kohya-ss/sd-scripts) implementation.

<img width="2554" height="1234" alt="image" src="https://github.com/user-attachments/assets/cb5ff930-ce8c-49d6-a77a-3da393fe719d" />


## Prerequisites

- **Python 3.10+** (Python 3.12 recommended)
- **Node.js** (Required for the Web UI)
- **CUDA fitting your system** (CUDA 12.7+ recommended)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Anima-Standalone-Trainer.git
cd Anima-Standalone-Trainer
```

### 2. Set up the environment

Run the provided setup script for your operating system:

**Windows:**
```powershell
.\setup_env.bat
```

*This will create a virtual environment (`venv`), install all Python dependencies (assuming you have met the prereqisites), and set up the Web UI.*

Depends on your system, you may want to install a specific version of Pytorch with CUDA.

## Launching the UI

To start the training server and open the web interface:
```bash
.\start_gui.bat
```
Once launched, open your browser to: `http://localhost:3000`


Some features and settings from sd-scripts may not be available or working properly at the momment.

Multi-GPU and Linux are not tested.

Built and tested on Windows 11, 1x RTX 3090, CUDA 13.1, Pytorch 2.10 


