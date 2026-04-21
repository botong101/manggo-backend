#!/usr/bin/env python
"""
Quick test script to verify anonymous prediction works
"""
import requests
import os

# Test the local server
url = "http://127.0.0.1:8000/api/predict/"

# Create a dummy image file for testing
test_image_path = "test_image.jpg"

# Check if test image exists, otherwise create a simple one
if not os.path.exists(test_image_path):
    from PIL import Image
    import numpy as np
    # Create a simple green image (simulating a leaf)
    img_array = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
    img_array[:, :, 1] = 150  # Make it greenish
    img = Image.fromarray(img_array)
    img.save(test_image_path)
    print(f"Created test image: {test_image_path}")

# Test without authentication
print("\n===== Testing without authentication =====")
with open(test_image_path, 'rb') as img:
    files = {'image': img}
    data = {
        'detection_type': 'leaf',
        'preview_only': 'true'
    }
    
    response = requests.post(url, files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

print("\n✅ Test complete!")
