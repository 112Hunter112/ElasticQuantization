import requests
import numpy as np

BASE_URL = "http://localhost:5000"

def main():
    # 1. Check Health
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Service Status: {resp.json()}")
    except Exception as e:
        print(f"Service unreachable: {e}")
        return

    # 2. Simulate Data
    # History: 1 sequence of 10 events, each with 768 dimensions
    history_vectors = np.random.randn(1, 10, 768).tolist()
    history_times = np.linspace(0, 10, 10).reshape(1, 10).tolist()
    target_time = 11.0
    
    payload = {
        "vectors": history_vectors,
        "timestamps": history_times,
        "target_time": target_time
    }
    
    print(f"\nSending request to Neural ODE...")
    
    # 3. Call Predict
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    
    if resp.status_code == 200:
        data = resp.json()
        print("\n=== Prediction Result ===")
        print(f"Backend:         {data.get('backend')}")
        print(f"Uncertainty:     {data.get('uncertainty'):.6f}")
        print(f"Anomaly Score:   {data.get('anomaly_score'):.6f}")
        print(f"Is Anomalous:    {data.get('is_anomalous')}")
    else:
        print(f"Error: {resp.text}")

if __name__ == "__main__":
    main()