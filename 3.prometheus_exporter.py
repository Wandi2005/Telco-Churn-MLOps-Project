from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
import random
from datetime import datetime

# Custom metrics untuk ML monitoring
class ModelMetricsExporter:
    def __init__(self):
        # Model performance metrics
        self.model_accuracy = Gauge('model_accuracy_gauge', 'Current model accuracy')
        self.model_precision = Gauge('model_precision_gauge', 'Current model precision')
        self.model_recall = Gauge('model_recall_gauge', 'Current model recall')
        self.model_f1 = Gauge('model_f1_score_gauge', 'Current model F1 score')
        self.model_roc_auc = Gauge('model_roc_auc_gauge', 'Current model ROC-AUC')
        
        # Data quality metrics
        self.missing_values = Gauge('missing_values_count', 'Number of missing values')
        self.data_drift = Gauge('data_drift_score', 'Data drift detection score')
        
        # System metrics
        self.prediction_queue_size = Gauge('prediction_queue_size', 'Predictions in queue')
        self.active_models = Gauge('active_models_count', 'Number of active models')
        self.api_response_time = Histogram('api_response_time_seconds', 'API response time')
        
        # Business metrics
        self.churn_rate = Gauge('customer_churn_rate', 'Customer churn rate percentage')
        self.total_predictions = Counter('total_predictions_made', 'Total predictions')
        self.churn_predictions = Counter('churn_predictions_total', 'Churn predictions', ['result'])
        
        # Username tracking (sesuai requirement)
        self.username = 'wandi_filemon_hotmartua_sianturi_7cZX'
        
    def update_metrics(self):
        """Update metrics dengan simulated data"""
        # Model performance
        self.model_accuracy.set(0.80 + random.uniform(-0.05, 0.05))
        self.model_precision.set(0.75 + random.uniform(-0.05, 0.05))
        self.model_recall.set(0.70 + random.uniform(-0.05, 0.05))
        self.model_f1.set(0.72 + random.uniform(-0.05, 0.05))
        self.model_roc_auc.set(0.85 + random.uniform(-0.03, 0.03))
        
        # Data quality
        self.missing_values.set(0)
        self.data_drift.set(random.uniform(0.01, 0.1))
        
        # System metrics
        self.prediction_queue_size.set(random.randint(0, 10))
        self.active_models.set(1)
        
        # Business metrics
        self.churn_rate.set(26.5 + random.uniform(-2, 2))
        self.total_predictions.inc()
        
        # Churn predictions
        if random.random() > 0.7:
            self.churn_predictions.labels(result='churn').inc()
        else:
            self.churn_predictions.labels(result='no_churn').inc()
        
        # API response time
        response_time = random.uniform(0.01, 0.1)
        self.api_response_time.observe(response_time)
        
    def start_server(self, port=8001):
        """Start Prometheus exporter"""
        start_http_server(port)
        print(f"📊 Prometheus Exporter started on port {port}")
        print(f"👤 Username: {self.username}")
        print(f"📈 Metrics available at: http://localhost:{port}/metrics")
        
        while True:
            self.update_metrics()
            time.sleep(10)

if __name__ == '__main__':
    exporter = ModelMetricsExporter()
    print("🚀 Starting Prometheus Metrics Exporter...")
    exporter.start_server(8001)