import random
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import time
import os
import logging
import threading 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Get app type from environment (hpa or combined)
app_type = os.environ.get('APP_TYPE', 'product-app') # Used in labels

# Initialize metrics exporter
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0', app_type=app_type)

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

# Counter for HTTP requests *successfully handled* by this Flask app
http_requests_total = Counter('app_http_requests_total', 'Total HTTP requests completed by Flask',
                            labelnames=['app', 'status_code'])

# In-memory storage for products with lock
products_lock = threading.Lock()
products = []

# Request tracking middleware
@app.before_request
def before_request_timing(): # Renamed to avoid potential conflicts
    request.start_time = time.time()

@app.after_request
def after_request_metrics(response): # Renamed to avoid potential conflicts
    # This counter only increments if the request *completes* within Flask
    http_requests_total.labels(app=app_type, status_code=str(response.status_code)).inc()
    return response

@app.route('/product/create', methods=['POST'])
def create_product():
    start_time = time.time() # Track latency for this endpoint too
    endpoint_path = '/product/create'
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

        name = data['name']
        description = data['description']
        price = data['price']

        with products_lock:
            product_id = len(products) + 1
            product = { 'id': product_id, 'name': name, 'description': description, 'price': price }
            products.append(product)
            products_count.labels(app=app_type).set(len(products))

        logger.info(f'{endpoint_path}: Product created: {product}')
        product_creation.labels(app=app_type).inc()
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({'message': 'Product created successfully', 'product': product}), 201

    except Exception as e: # General exception handling
        product_creation_errors.labels(app=app_type).inc()
        logger.exception(f'{endpoint_path}: Error creating product: {e}')
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        # Note: A 500 status code will be recorded by after_request_metrics
        return jsonify({'error': 'Failed to create product'}), 500


@app.route('/product/list', methods=['GET'])
def list_products():
    start_time = time.time()
    endpoint_path = '/product/list'
    response = None
    status_code = 500 # Default to error
    try:
        product_list_requests.labels(app=app_type).inc()

        # Simulate potential slowness for demonstration if needed
        # time.sleep(random.uniform(0.1, 0.5))

        with products_lock:
            # Creating a copy inside the lock is safer
            products_copy = list(products)

        response = jsonify({"products": products_copy})
        status_code = 200

    except Exception as e:
        logger.exception(f'{endpoint_path}: Error listing products: {e}')
        response = jsonify({'error': 'Failed to list products'})
        status_code = 500
    finally:
        # Ensure latency is always recorded
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        # Manually create Flask response object if needed for status code
        if response is None:
             response = jsonify({'error': 'An unexpected error occurred during final processing'})
             status_code = 500
        response.status_code = status_code

    return response


@app.route('/product/<id>', methods=['GET'])
def get_product(id):
    start_time = time.time()
    endpoint_path = f'/product/{id}' # Use dynamic path for potential labeling, though often grouped
    generic_endpoint_path = '/product/<id>' # For latency metric grouping

    try:
        product_id = int(id)
        product_copy = None
        with products_lock:
            if 0 < product_id <= len(products):
                # Ensure deep copy if products contained mutable objects
                product_copy = products[product_id-1].copy()

        if product_copy:
             latency = time.time() - start_time
             request_latency.labels(app=app_type, endpoint=generic_endpoint_path).observe(latency)
             return jsonify({"product": product_copy}), 200
        else:
            latency = time.time() - start_time
            request_latency.labels(app=app_type, endpoint=generic_endpoint_path).observe(latency)
            return jsonify({"error": "Product not found"}), 404

    except ValueError:
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=generic_endpoint_path).observe(latency)
        return jsonify({"error": "Invalid product ID format"}), 400
    except Exception as e: # General error handling
        logger.exception(f"{endpoint_path}: Error getting product: {e}")
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=generic_endpoint_path).observe(latency)
        return jsonify({"error": "Internal Server Error retrieving product"}), 500


@app.route('/load', methods=['GET'])
def generate_load():
    start_time = time.time()
    endpoint_path = '/load'
    try:
        # Use lower defaults if not specified
        duration = int(request.args.get('duration', 1))
        intensity = int(request.args.get('intensity', 5)) # Lower default intensity

        # Clamp duration/intensity to prevent accidental DoS? Optional.
        duration = min(duration, 15) # Max 15 seconds duration via param
        intensity = min(intensity, 20) # Max 20 intensity via param

        logger.info(f"{endpoint_path}: Generating load duration={duration}s, intensity={intensity}")

        loop_start_time = time.time()
        end_time = loop_start_time + duration

        # This loop can still cause the Gunicorn worker to exceed its timeout
        while time.time() < end_time:
            # Reduced inner loop iterations
            for _ in range(intensity * 500): # Reduced calculation load
                result = sum(random.random() for _ in range(500)) # Reduced inner sum
            # Short sleep to yield control, preventing 100% CPU hogging within the loop
            time.sleep(0.01)

        actual_duration = time.time() - loop_start_time
        logger.info(f"{endpoint_path}: Load generation completed in {actual_duration:.2f}s")
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({'message': f'Load generated for ~{duration}s with intensity {intensity}. Actual loop duration: {actual_duration:.2f}s'}), 200

    except ValueError:
         latency = time.time() - start_time
         request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
         return jsonify({'error': 'Invalid duration or intensity parameters. Must be integers.'}), 400
    except Exception as e: # General exception handling
        logger.error(f"{endpoint_path}: Error generating load: {e}", exc_info=True)
        latency = time.time() - start_time
        request_latency.labels(app=app_type, endpoint=endpoint_path).observe(latency)
        return jsonify({'error': 'Failed to generate load'}), 500


@app.route('/health', methods=['GET'])
def health():

    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # This block is for local development testing ONLY
    print(f"--- Starting {app_type} in DEVELOPMENT mode ---")
    print("--- WARNING: Not suitable for production or load testing! ---")
    print("--- Recommend running with Gunicorn: ---")
    print("gunicorn -w 4 -b 0.0.0.0:5000 --timeout 10 app:app") # Show recommended command with timeout
    # Run Flask's development server (less efficient, different concurrency model)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) # Use debug=True carefully