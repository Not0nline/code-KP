@app.route('/api/inject_data', methods=['POST'])
def inject_data():
    """Manually inject a single data point into the traffic buffer."""
    global traffic_data
    try:
        data_point = request.get_json()
        if not data_point or 'timestamp' not in data_point:
            return jsonify({'success': False, 'error': 'Invalid data point format'}), 400
        
        # Simple validation
        required_keys = ['timestamp', 'traffic', 'cpu_utilization', 'replicas']
        if not all(key in data_point for key in required_keys):
            return jsonify({'success': False, 'error': f'Missing one of required keys: {required_keys}'}), 400
            
        traffic_data.append(data_point)
        
        # Optional: log every Nth injection to avoid spamming logs
        if len(traffic_data) % 100 == 0:
            logger.info(f"Data injected. Current buffer size: {len(traffic_data)}")
            
        return jsonify({'success': True, 'buffer_size': len(traffic_data)}), 200
        
    except Exception as e:
        logger.error(f"Failed to inject data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
