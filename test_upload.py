import requests

url = "http://localhost:8000/upload-pdf"
filename = "Jeffrey_Lebowski_CV.pdf"

print(f"📤 Uploading {filename}...")
print(f"🎯 URL: {url}")

with open(filename, "rb") as f:
    response = requests.post(url, files={"file": f})
    print(f"📊 Status: {response.status_code}")
    print(f"📝 Response: {response.text}")
