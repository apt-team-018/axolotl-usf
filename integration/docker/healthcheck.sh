#!/bin/bash
# Health check script for Axolotl USF container
# Returns 0 if healthy, 1 if unhealthy

set -e

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Check if integration layer is accessible
if [ ! -d "/workspace/integration" ]; then
    echo "ERROR: Integration layer not found"
    exit 1
fi

# Check if we can import key modules
python3 -c "
import sys
try:
    # Test basic imports
    import torch
    import axolotl
    from integration.train_wrapper import TrainingWrapper

    # Check GPU availability (if CUDA is available)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print('WARNING: CUDA available but no devices found', file=sys.stderr)
            sys.exit(1)

    print('✓ Health check passed')
    sys.exit(0)

except ImportError as e:
    print(f'ERROR: Import failed: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'ERROR: Health check failed: {e}', file=sys.stderr)
    sys.exit(1)
"

exit $?
