import os

def setup_tts_license():
    """Set up TTS license agreement."""
    print("Setting up TTS license...")
    
    os.environ["COQUI_TTS_AGREED"] = "1"
    os.environ["TTS_LICENSE_AGREEMENT_SIGNED"] = "1"
    print(f"COQUI_TTS_AGREED env var: {os.getenv('COQUI_TTS_AGREED')}")
    
    license_path = '/app/.local/share/tts/license_agreement_signed'
    os.makedirs(os.path.dirname(license_path), exist_ok=True)
    with open(license_path, 'w') as f:
        f.write('True')
    print(f"License file created at: {license_path}")
    print(f"License file exists: {os.path.exists(license_path)}")
    print(f"License file content: {open(license_path).read()}")