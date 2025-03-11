#!/bin/bash

# Portfolio Optimization Project Initialization Script
# This script sets up the project structure and initializes required files

# Set color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Portfolio Optimization Project...${NC}"

# Create directory structure
echo "Creating directory structure..."
mkdir -p config utils src tests data/raw data/processed dashboard logs notebooks

# Create __init__.py files
echo "Creating Python package files..."
touch config/__init__.py utils/__init__.py src/__init__.py tests/__init__.py
touch src/__init__.py

# Create .gitkeep files for empty directories
echo "Creating .gitkeep files for empty directories..."
touch data/raw/.gitkeep data/processed/.gitkeep dashboard/.gitkeep logs/.gitkeep

# Create environment file from template
echo "Creating environment file from template..."
if [ -f .env.template ]; then
    cp .env.template .env
    echo -e "${YELLOW}Remember to update your .env file with your API keys and settings${NC}"
else
    echo -e "${RED}Error: .env.template not found${NC}"
    exit 1
fi

# Check if Docker is installed
echo "Checking for Docker installation..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker is installed${NC}"
else
    echo -e "${YELLOW}Docker not found. You can still run the project with Python directly.${NC}"
fi

# Check if Python is installed
echo "Checking for Python installation..."
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}Python is installed${NC}"
    
    # Create virtual environment
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    # Install requirements
    echo "Installing requirements..."
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    else
        echo -e "${RED}Error: requirements.txt not found${NC}"
        exit 1
    fi
else
    echo -e "${RED}Python not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

echo -e "${GREEN}Project setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update your .env file with your API keys and settings"
echo "2. Run the pipeline with: python main.py"
echo "3. Or run individual components with: python -m src.data_collection"
echo ""
echo -e "${GREEN}Happy optimizing!${NC}"