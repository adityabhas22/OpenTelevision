#!/usr/bin/env python3
"""
RUKA Hand Teleoperation via Apple Vision Pro

This script provides the entry point for teleoperating RUKA hands using
Apple Vision Pro hand tracking through OpenTeleVision.

Usage:
    # Right hand only (default)
    python teleop_ruka.py --hand-type right
    
    # Both hands
    python teleop_ruka.py --hand-type both
    
    # Test mode (stream only, no motor control)
    python teleop_ruka.py --stream-only
    
Requirements:
    - RUKA hand connected via USB
    - SSL certificates (cert.pem, key.pem) for HTTPS
    - Apple Vision Pro on same network
    
Setup:
    1. Generate SSL certs: openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
    2. Run motor calibration: cd RUKA && python calibrate_motors.py --hand-type right
    3. Start teleop: python teleop_ruka.py --hand-type right
    4. Open Safari on AVP: https://<your-pc-ip>:8012
"""

import sys
from pathlib import Path

# Add RUKA to path
SCRIPT_DIR = Path(__file__).parent
RUKA_PATH = SCRIPT_DIR.parent / "RUKA"
sys.path.insert(0, str(RUKA_PATH))

from ruka_hand.teleoperation.avp_teleoperator import AVPTeleoperator, main

if __name__ == "__main__":
    main()
