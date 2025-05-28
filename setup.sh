#!/bin/bash
# DomainTest Setup Script
# This script sets up the DomainBed dependency for the DomainTest framework

echo "Setting up DomainTest environment..."

# Check if DomainBed directory exists
if [ ! -d "DomainBed" ]; then
    echo "Cloning DomainBed repository..."
    git clone https://github.com/facebookresearch/DomainBed.git
    
    if [ $? -eq 0 ]; then
        echo "✓ DomainBed cloned successfully"
    else
        echo "✗ Failed to clone DomainBed"
        exit 1
    fi
else
    echo "✓ DomainBed directory already exists"
fi

# Install DomainBed requirements
if [ -f "DomainBed/requirements.txt" ]; then
    echo "Installing DomainBed requirements..."
    pip install -r DomainBed/requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✓ DomainBed requirements installed"
    else
        echo "✗ Failed to install DomainBed requirements"
        exit 1
    fi
fi

# Install DomainTest requirements
if [ -f "requirements.txt" ]; then
    echo "Installing DomainTest requirements..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✓ DomainTest requirements installed"
    else
        echo "✗ Failed to install DomainTest requirements"
        exit 1
    fi
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "You can now run experiments:"
echo "  python main.py                    # Run with default config"
echo "  python main.py --summary          # View experiment summary"
echo "  python batch_experiments_v2.py    # Run batch experiments"
echo "  python vgg16_domain_test.py --dataset ColoredMNIST --test_env 0"
echo ""
