import random
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import os
import logging
import threading 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Get app type from environment (hpa or combined)
app_type = os.environ.get('APP_TYPE', 'product-app')

# Custom metrics
product_creation = Counter('product_creation_total', 'Total number of products created',
                           labelnames=['app'])
product_creation_errors = Counter('product_creation_errors_total', 'Total number of product creation errors',
                                 labelnames=['app'])
product_list_requests = Counter('product_list_requests_total', 'Total number of product list requests',
                               labelnames=['app'])
request_latency = Histogram('request_latency_seconds', 'Request latency in seconds',
                          labelnames=['app', 'endpoint'])
products_count = Gauge('products_count', 'Current number of products',
                      labelnames=['app'])
http_requests_total = Counter('app_http_requests_total', 'Total HTTP requests completed by Flask',
                            labelnames=['app', 'status_code'])
rejected_requests = Counter('rejected_requests_total', 'Requests rejected due to overload',
                           labelnames=['app'])

# In-memory storage for products with lock
products_lock = threading.Lock()
products = []

# Track concurrent requests
concurrent_requests = 0
# Tighter concurrency cap so overload conditions surface quickly under stress
max_concurrent_requests = max(1, int(os.environ.get('MAX_CONCURRENT_REQUESTS', '120')))  # Reject if more than this
request_lock = threading.Lock()

# Ensure counters exist at startup so Prometheus scrapes have stable series even when idle
def _initialize_metrics_baseline():
    try:
        for code in ('200', '201', '400', '404', '500', '503'):
            # inc(0) registers the timeseries without changing its value
            http_requests_total.labels(app=app_type, status_code=code).inc(0)
        rejected_requests.labels(app=app_type).inc(0)
        products_count.labels(app=app_type).set(0)
    except Exception:
        # Best-effort; don't block app startup
        pass

_initialize_metrics_baseline()

@app.before_request
def before_request_timing():
    global concurrent_requests
    
    # Always allow fast paths for health and metrics; don't throttle or count concurrency
    if request.path in ('/metrics', '/health'):
        request.start_time = time.time()
        return

    # Check if we're overloaded for application endpoints only
    with request_lock:
        if concurrent_requests >= max_concurrent_requests:
            rejected_requests.labels(app=app_type).inc()
            logger.warning(f"Rejecting request - too many concurrent requests: {concurrent_requests}")
            # Return 503 Service Unavailable immediately
            response = jsonify({'error': 'Service overloaded, please retry later'})
            response.status_code = 503
            return response
        concurrent_requests += 1
    
    request.start_time = time.time()

@app.after_request
def after_request_metrics(response):
    global concurrent_requests
    
    # Don't track concurrency or metrics for health/metrics paths
    if request.path not in ('/metrics', '/health'):
        # Decrement concurrent requests
        with request_lock:
            concurrent_requests = max(0, concurrent_requests - 1)
        # Count only application requests in custom metric
        http_requests_total.labels(app=app_type, status_code=str(response.status_code)).inc()
    return response

@app.route('/product/create', methods=['POST'])
def create_product():
    start_time = time.time()
    endpoint_path = '/product/create'
    
    # Simulate some processing time
    time.sleep(random.uniform(0.01, 0.05))
    
    try:
        data = request.get_json(silent=True)
        if not data:
            logger.error(f'{endpoint_path}: Empty request body')
            return jsonify({'error': 'Empty request body'}), 400

        required_fields = ['name', 'description', 'price']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f'{endpoint_path}: Missing fields: {", ".join(missing_fields)}')
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        # Removed artificial random failures to keep behavior deterministic in tests
        # if random.random() < 0.03:
        #     product_creation_errors.labels(app=app_type).inc()
        #     return jsonify({'error': 'Random failure for testing'}), 500

        name = data['name']
        description = data['description']
        price = data['price']

        with products_lock:
            product_id = len(products) + 1
            product = {'id': product_id, 'name': name, 'description': description, 'price': price}
            products.append(product)
            products_count.labels(app=app_type).set(len(products))

        logger.info(f'{endpoint_path}: Product created: {product}')
        product_creation.labels(app=app_type).inc()
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({'message': 'Product created successfully', 'product': product}), 201

    except Exception as e:
        product_creation_errors.labels(app=app_type).inc()
        logger.exception(f'{endpoint_path}: Error creating product: {e}')
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({'error': 'Failed to create product'}), 500

@app.route('/product/list', methods=['GET'])
def list_products():
    start_time = time.time()
    endpoint_path = '/product/list'
    
    # Simulate some processing time
    time.sleep(random.uniform(0.01, 0.03))
    
    try:
        product_list_requests.labels(app=app_type).inc()
        
        # Removed artificial random failures to keep behavior deterministic in tests
        # if random.random() < 0.00:
        #     raise Exception("Random failure for testing")
        
        with products_lock:
            products_copy = list(products)
        
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({"products": products_copy}), 200
        
    except Exception as e:
        logger.exception(f'{endpoint_path}: Error listing products: {e}')
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({'error': 'Failed to list products'}), 500

@app.route('/product/<id>', methods=['GET'])
def get_product(id):
    start_time = time.time()
    generic_endpoint_path = '/product/<id>'
    
    try:
        product_id = int(id)
        
        # Simulate some processing time
        time.sleep(random.uniform(0.01, 0.02))
        
        with products_lock:
            if 0 < product_id <= len(products):
                product_copy = products[product_id-1].copy()
            else:
                product_copy = None
        
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=generic_endpoint_path).observe(latency)
        
        if product_copy:
            return jsonify({"product": product_copy}), 200
        else:
            return jsonify({"error": "Product not found"}), 404
            
    except ValueError:
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=generic_endpoint_path).observe(latency)
        return jsonify({"error": "Invalid product ID format"}), 400
    except Exception as e:
        logger.exception(f"Error getting product: {e}")
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=generic_endpoint_path).observe(latency)
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/load', methods=['GET'])
def generate_load():
    """Non-blocking load generation endpoint"""
    start_time = time.time()
    endpoint_path = '/load'
    
    try:
        # Accept fractional seconds and allow very short bursts
        try:
            duration = float(request.args.get('duration', 1))
        except Exception:
            duration = 1.0
        duration = max(0.05, min(duration, 5.0))  # Clamp to [0.05, 5.0] seconds
        # Allow much higher intensity to generate aggressive CPU spikes
        intensity = min(int(request.args.get('intensity', 5)), 500)  # Max 500 intensity

        logger.info(f"{endpoint_path}: Generating load duration={duration}s, intensity={intensity}")
        
        # Non-blocking CPU intensive work
        end_time = time.time() + duration
        iterations = 0
        
        while time.time() < end_time:
            # Do small chunks of work
            # Increase work proportional to intensity to amplify CPU usage
            for _ in range(intensity * 200):
                _ = sum(random.random() for _ in range(100))
                iterations += 1
            
            # Yield to other requests
            if iterations % 1000 == 0:
                time.sleep(0.001)  # Very brief yield
        
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({'message': f'Load generated for {duration}s with intensity {intensity}'}), 200
        
    except Exception as e:
        logger.error(f"{endpoint_path}: Error generating load: {e}")
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({'error': 'Failed to generate load'}), 500

@app.route('/health', methods=['GET'])
def health():
    # Quick health check - don't count against concurrent requests
    return jsonify({"status": "healthy"}), 200

@app.route('/reset_data', methods=['POST'])
def reset_data():
    """Reset all product data - clear the in-memory database"""
    global concurrent_requests
    start_time = time.time()
    endpoint_path = '/reset_data'
    
    try:
        with products_lock:
            products_before = len(products)
            products.clear()
            products_count.labels(app=app_type).set(0)
        
        with request_lock:
            concurrent_requests = 0
        
        logger.info(f'{endpoint_path}: Database reset - cleared {products_before} products')
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        
        return jsonify({
            'message': 'Database reset successfully',
            'products_cleared': products_before,
            'current_products': 0
        }), 200
        
    except Exception as e:
        logger.exception(f'{endpoint_path}: Error resetting database: {e}')
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({'error': 'Failed to reset database'}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    # Expose Prometheus metrics
    from flask import Response
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/products', methods=['GET'])
def products_endpoint():
    """Alias for /product/list for compatibility"""
    return list_products()

if __name__ == '__main__':
    print(f"--- Starting {app_type} in DEVELOPMENT mode ---")
    print("--- WARNING: Not suitable for production! ---")
    print("--- Use Gunicorn: gunicorn -w 1 -b 0.0.0.0:5000 --timeout 10 app:app")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)