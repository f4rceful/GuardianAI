import requests
import json

def test_api():
    url = "http://127.0.0.1:8550/predict"
    
    # Payload similar to what Android will send
    payload = {
        "text": "Мама, срочно переведи 5000р на карту, у меня проблемы!",
        "context": []
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success! Response:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            if data['is_scam']:
                print("Correctly identified as SCAM.")
            else:
                print("WARNING: Identified as SAFE (Check model threshold?).")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_api()
