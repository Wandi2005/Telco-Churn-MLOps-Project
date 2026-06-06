from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram, Gauge, generate_latest
import time
import os
import threading

# Initialize Flask app
app = Flask(__name__)

# ===== PROMETHEUS METRICS (WAJIB ADA) =====
# 1. Counter untuk total prediksi
PREDICTION_COUNTER = Counter(
    'predictions_total', 
    'Total number of predictions made',
    ['churn']  # label: 'yes' atau 'no'
)

# 2. Histogram untuk latency
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing predictions',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# 3. Gauge untuk akurasi model
MODEL_ACCURACY = Gauge(
    'model_accuracy_gauge',
    'Current model accuracy score'
)

# 4. Gauge untuk koneksi aktif
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections to the API'
)

# 5. Gauge untuk churn rate
CHURN_RATE = Gauge(
    'churn_rate',
    'Churn rate percentage from recent predictions'
)

# Store predictions for monitoring
predictions_log = []

# ===== LOAD MODEL & PREPROCESSING =====
print("🔄 Loading model and preprocessors...")
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    # Set akurasi model asli (ganti dengan akurasi modelmu yang sebenarnya)
    MODEL_ACCURACY.set(0.85)  # Contoh: 85% accuracy
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model not found - {e}")
    print("⚠️ Using dummy model for testing...")
    model = None
    scaler = None
    label_encoders = None
    feature_names = None
    MODEL_ACCURACY.set(0.0)

# ===== API ENDPOINTS =====

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint untuk monitoring service status
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Telco Churn Prediction API',
        'username': 'wandi_filemon_hotmartua_sianturi_7cZX',
        'timestamp': time.time()
    }), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Prometheus metrics endpoint - wajib ada!
    """
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/predict', methods=['POST'])
@PREDICTION_LATENCY.time()  # Decorator untuk auto-track latency
def predict():
    """
    Single prediction endpoint dengan monitoring metrics
    """
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()  # Naikkan counter koneksi aktif
    
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            ACTIVE_CONNECTIONS.dec()
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Predict dengan model ASLI atau dummy (TANPA RANDOM!)
        if model is not None and scaler is not None:
            # Preprocessing
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
        else:
            # Dummy prediction: KONSISTEN (tidak random!)
            # Metrics tetap tercatat dari aktivitas API nyata
            prediction = 0  # Selalu return "No Churn" untuk dummy
            prediction_proba = [0.5, 0.5]
        
        # Log prediction untuk monitoring
        predictions_log.append({
            'timestamp': time.time(),
            'prediction': int(prediction),
            'probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.5
        })
        
        # Update metrics ASLI (bukan random!)
        churn_label = 'yes' if prediction == 1 else 'no'
        PREDICTION_COUNTER.labels(churn=churn_label).inc()
        
        # Calculate churn rate dari 100 prediksi terakhir
        if len(predictions_log) >= 10:
            recent_churns = sum(1 for p in predictions_log[-100:] if p['prediction'] == 1)
            churn_percentage = (recent_churns / min(len(predictions_log), 100)) * 100
            CHURN_RATE.set(churn_percentage)
        
        # Hitung latency ASLI
        latency = time.time() - start_time
        
        ACTIVE_CONNECTIONS.dec()  # Turunkan counter setelah selesai
        
        return jsonify({
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.5,
            'latency_seconds': latency,
            'username': 'wandi_filemon_hotmartua_sianturi_7cZX',
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        ACTIVE_CONNECTIONS.dec()
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint untuk multiple predictions
    """
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected list of data'}), 400
        
        input_df = pd.DataFrame(data)
        
        # Predict dengan model ASLI atau dummy (TANPA RANDOM!)
        if model is not None and scaler is not None:
            input_scaled = scaler.transform(input_df)
            predictions = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
        else:
            # Dummy predictions: KONSISTEN (tidak random!)
            predictions = np.zeros(len(input_df), dtype=int)
            probabilities = np.array([[0.5, 0.5] for _ in range(len(input_df))])
        
        # Update metrics untuk setiap prediction
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'index': i,
                'prediction': 'Churn' if pred == 1 else 'No Churn',
                'probability': float(prob[1]) if len(prob) > 1 else 0.5
            })
            
            churn_label = 'yes' if pred == 1 else 'no'
            PREDICTION_COUNTER.labels(churn=churn_label).inc()
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'username': 'wandi_filemon_hotmartua_sianturi_7cZX',
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get model information
    """
    return jsonify({
        'model_type': type(model).__name__ if model else 'Dummy Model',
        'model_loaded': model is not None,
        'features': feature_names if feature_names else [],
        'username': 'wandi_filemon_hotmartua_sianturi_7cZX'
    }), 200

@app.route('/predictions_log', methods=['GET'])
def get_predictions_log():
    """
    Get recent predictions log
    """
    limit = request.args.get('limit', 100, type=int)
    return jsonify({
        'predictions': predictions_log[-limit:],
        'total': len(predictions_log),
        'username': 'wandi_filemon_hotmartua_sianturi_7cZX'
    }), 200

# ===== START SERVERS =====

def start_prometheus_server(port=8000):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"📊 Prometheus metrics server started on port {port}")

if __name__ == '__main__':
    print("="*60)
    print("🚀 TELCO CHURN PREDICTION API")
    print("="*60)
    print(f"📊 Metrics available at: http://localhost:8000/metrics")
    print(f"🏥 Health check at: http://localhost:5001/health")
    print(f"🔮 Predict at: http://localhost:5001/predict")
    print(f"📦 Batch predict at: http://localhost:5001/batch_predict")
    print("="*60)
    
    # Start Prometheus server di thread terpisah
    prometheus_thread = threading.Thread(
        target=start_prometheus_server,
        args=(8000,),
        daemon=True
    )
    prometheus_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)