from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram, Gauge, generate_latest
import time
import os

# Initialize Flask app
app = Flask(__name__)

# Prometheus Metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions', ['churn_result'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('model_accuracy_gauge', 'Model accuracy')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
CHURN_RATE = Gauge('churn_rate', 'Churn rate percentage')

# Load model dan preprocessing
print("🔄 Loading model and preprocessors...")
try:
    model = joblib.load('../Membangun_model/best_model.pkl')
    scaler = joblib.load('../Eksperimen_SML_Nama-siswa/scaler.pkl')
    label_encoders = joblib.load('../Eksperimen_SML_nama-siswa/label_encoders.pkl')
    feature_names = joblib.load('../Eksperimen_SML_nama-siswa/feature_names.pkl')
    print("✅ Model loaded successfully!")
except:
    print("⚠️ Model not found. Using dummy model for testing...")
    model = None
    scaler = None
    feature_names = None

# Store predictions for monitoring
predictions_log = []

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Telco Churn Prediction API',
        'username': 'wandi_filemon_hotmartua_sianturi_7cZX'
    }), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/predict', methods=['POST'])
@PREDICTION_LATENCY.time()
def predict():
    """Single prediction endpoint"""
    try:
        ACTIVE_CONNECTIONS.inc()
        
        data = request.get_json()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Predict (dummy jika model tidak ada)
        if model is not None:
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
        else:
            prediction = np.random.randint(0, 2)
            prediction_proba = [0.5, 0.5]
        
        # Log prediction
        predictions_log.append({
            'timestamp': time.time(),
            'prediction': int(prediction),
            'probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.5
        })
        
        # Update metrics
        churn_label = 'yes' if prediction == 1 else 'no'
        PREDICTION_COUNTER.labels(churn=churn_label).inc()
        
        # Calculate churn rate (last 100 predictions)
        if len(predictions_log) >= 10:
            recent_churns = sum(1 for p in predictions_log[-100:] if p['prediction'] == 1)
            CHURN_RATE.set(recent_churns / min(len(predictions_log), 100) * 100)
        
        ACTIVE_CONNECTIONS.dec()
        
        return jsonify({
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.5,
            'username': 'wandi_filemon_hotmartua_sianturi_7cZX',
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        ACTIVE_CONNECTIONS.dec()
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        input_df = pd.DataFrame(data)
        
        if model is not None:
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)
        else:
            predictions = np.random.randint(0, 2, len(input_df))
            probabilities = np.array([[0.5, 0.5] for _ in range(len(input_df))])
        
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
            'username': 'wandi_filemon_hotmartua_sianturi_7cZX'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': type(model).__name__ if model else 'Dummy Model',
        'features': feature_names if feature_names else [],
        'username': 'wandi_filemon_hotmartua_sianturi_7cZX'
    }), 200

@app.route('/predictions_log', methods=['GET'])
def get_predictions_log():
    """Get recent predictions"""
    limit = request.args.get('limit', 100, type=int)
    return jsonify({
        'predictions': predictions_log[-limit:],
        'total': len(predictions_log),
        'username': 'wandi_filemon_hotmartua_sianturi_7cZX'
    }), 200

def start_prometheus_server(port=8000):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"📊 Prometheus metrics server started on port {port}")

if __name__ == '__main__':
    # Start Prometheus server
    import threading
    prometheus_thread = threading.Thread(
        target=start_prometheus_server,
        args=(8000,),
        daemon=True
    )
    prometheus_thread.start()
    
    # Start Flask app
    print("🚀 Starting Inference API on port 5001...")
    print(f"📊 Metrics available at: http://localhost:8000/metrics")
    print(f"🔮 Predictions available at: http://localhost:5001/predict")
    app.run(host='0.0.0.0', port=5001, debug=True)