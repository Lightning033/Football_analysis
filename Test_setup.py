# TEST SCRIPT - Run this to verify everything works
import sys
print("🔍 TESTING PYTHON SETUP...")
print("-" * 30)

try:
    import cv2
    print("✅ OpenCV:", cv2.__version__)
except ImportError:
    print("❌ OpenCV not installed - run: pip install opencv-python")

try:
    import numpy
    print("✅ NumPy:", numpy.__version__)
except ImportError:
    print("❌ NumPy not installed - run: pip install numpy")

try:
    import matplotlib
    print("✅ Matplotlib:", matplotlib.__version__)
except ImportError:
    print("❌ Matplotlib not installed - run: pip install matplotlib")

try:
    import ultralytics
    print("✅ Ultralytics (YOLOv8) ready")
except ImportError:
    print("❌ Ultralytics not installed - run: pip install ultralytics")

try:
    import pandas
    print("✅ Pandas:", pandas.__version__)
except ImportError:
    print("❌ Pandas not installed - run: pip install pandas")

try:
    import pytesseract
    print("✅ Pytesseract ready")
except ImportError:
    print("❌ Pytesseract not installed - run: pip install pytesseract")

print("\n🎯 TESTING CAMERA POSITIONING...")
print("-" * 30)

try:
    from camera_positioning import CameraPositioning
    positioning = CameraPositioning()
    positions = positioning.calculate_optimal_positions()
    print("✅ Camera positioning tool ready")
    print(f"✅ Found {len(positions)} optimal camera positions")
except Exception as e:
    print(f"❌ Camera positioning error: {e}")
    print("Make sure camera_positioning.py is in your project folder")

print("\n🎮 TESTING YOLO MODEL DOWNLOAD...")
print("-" * 30)

try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # This will download the model if not present
    print("✅ YOLOv8 model downloaded and ready")
except Exception as e:
    print(f"❌ YOLO model error: {e}")

print("\n🏁 TEST COMPLETE!")
print("=" * 40)

# Count successes
import importlib
packages = ['cv2', 'numpy', 'matplotlib', 'ultralytics', 'pandas', 'pytesseract']
successful = 0

for package in packages:
    try:
        importlib.import_module(package)
        successful += 1
    except ImportError:
        pass

print(f"📊 RESULTS: {successful}/{len(packages)} packages working")

if successful == len(packages):
    print("🎉 SUCCESS! Your environment is ready for football analytics!")
    print("\nNext steps:")
    print("1. Run: python camera_positioning.py")
    print("2. Set up your GoPro cameras using the generated diagram")
    print("3. Record test footage and start development!")
else:
    print(f"⚠️  {len(packages) - successful} packages need to be installed")
    print("Install missing packages using the commands shown above")
