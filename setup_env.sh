#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

echo "----------------------------------------------------------------------"
echo "Checking Prerequisites..."
echo "----------------------------------------------------------------------"

if ! command -v node &> /dev/null
then
    echo ""
    echo "[ERROR] Node.js is not installed!"
    echo "Node.js is required for the Training UI."
    echo "Please install it using your package manager (e.g., sudo apt install nodejs)"
    echo "or download it from: https://nodejs.org/"
    echo ""
    exit 1
fi

echo "Node.js detected."
echo ""

if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
else
    echo "Venv already exists."
fi

source venv/bin/activate

echo "----------------------------------------------------------------------"
echo "Installing requirements from requirements.txt..."
echo "----------------------------------------------------------------------"
pip install -r requirements.txt

echo ""
echo "----------------------------------------------------------------------"
echo "Installing UI dependencies (npm install)..."
echo "----------------------------------------------------------------------"
cd training-ui
npm install
cd ..

echo ""
echo "----------------------------------------------------------------------"
echo "Installation Complete!"
echo "----------------------------------------------------------------------"
