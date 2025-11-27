
@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    """Manually clear all data buffers."""
    global traffic_data, training_dataset, predictions_history, model_mse_values, is_model_trained
    traffic_data.clear()
    training_dataset = []
    is_model_trained = False
    
    # Reset predictions and MSE
    for model_name in predictions_history:
        predictions_history[model_name] = []
    for model_name in model_mse_values:
        model_mse_values[model_name] = float('inf')
        
    logger.info("All data buffers, predictions, and model states cleared via API.")
    return jsonify({'success': True, 'message': 'All data cleared.'})

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
            # For HPA data that uses 'cpu'
            if 'cpu' in data_point and 'cpu_utilization' not in data_point:
                data_point['cpu_utilization'] = data_point['cpu']
            else:
                return jsonify({'success': False, 'error': f'Missing one of required keys: {required_keys}'}), 400
            
        traffic_data.append(data_point)
        
        # Optional: log every Nth injection to avoid spamming logs
        if len(traffic_data) % 200 == 0:
            logger.info(f"Data injected. Current buffer size: {len(traffic_data)}")
            
        return jsonify({'success': True, 'buffer_size': len(traffic_data)}), 200
        
    except Exception as e:
        logger.error(f"Failed to inject data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
