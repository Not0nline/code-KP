
@app.route('/api/load_and_train', methods=['POST'])
def load_and_train():
    """
    Atomically loads a baseline dataset and triggers training
    to avoid race conditions between separate API calls.
    """
    global traffic_data, training_dataset, data_collection_complete
    
    try:
        # 1. Load the baseline data (logic from load_baseline_dataset)
        from baseline_datasets import BaselineDatasetManager
        
        data = request.get_json()
        scenario = data.get('scenario', '').lower()
        
        if scenario not in ['low', 'medium', 'high']:
            return jsonify({'success': False, 'error': f"Invalid scenario: {scenario}"}), 400
        
        logger.info(f" atomically loading and training for scenario: {scenario}")
        
        manager = BaselineDatasetManager()
        baseline_data = manager.load_baseline(scenario)
        
        traffic_data.clear()
        traffic_data.extend(baseline_data['traffic_data'])
        training_dataset = list(traffic_data)
        data_collection_complete = True
        logger.info(f"Data loaded and buffers populated with {len(training_dataset)} points.")

        # 2. Train the models (logic from train_all_models)
        from model_variants import initialize_model_registry
        
        if len(training_dataset) < 200:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for training after load',
                'current_points': len(training_dataset),
                'required_points': 200
            }), 400
            
        registry = initialize_model_registry()
        logger.info(f"Training all models with {len(training_dataset)} data points...")
        results = registry.train_all_models(training_dataset)
        registry.save_all_models()
        
        successful = sum(1 for r in results.values() if r.get('success'))
        
        return jsonify({
            'success': True,
            'operation': 'load_and_train',
            'scenario': scenario,
            'data_points_loaded': len(training_dataset),
            'models_trained': successful,
            'total_models': len(results),
            'results': results
        })

    except Exception as e:
        logger.error(f"Failed during load_and_train: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
