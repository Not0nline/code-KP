#!/bin/bash
# Build and Deploy 11-Model Predictive Scaler

echo "🚀 Building 11-Model Predictive Scaler Docker Image..."
echo "📊 Models: GRU, LSTM, Holt-Winters, XGBoost, LightGBM, StatuScale, ARIMA, CNN, Autoencoder, Prophet, Ensemble"

# Build Docker image with updated tag
docker build -t 4dri4l/predictive-scaler:v3.0-11models .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Docker build successful!"
    
    # Push to registry
    echo "📤 Pushing to Docker Hub..."
    docker push 4dri4l/predictive-scaler:v3.0-11models
    
    if [ $? -eq 0 ]; then
        echo "✅ Push successful!"
        echo "🎯 Image ready: 4dri4l/predictive-scaler:v3.0-11models"
        
        echo ""
        echo "📋 NEXT STEPS:"
        echo "1. Update deployment YAML to use new image"
        echo "2. Deploy to Kubernetes cluster"
        echo "3. Verify all 11 models are working"
        echo "4. Run comprehensive tests"
        
        echo ""
        echo "🔧 UPDATE COMMAND:"
        echo "kubectl set image deployment/predictive-scaler predictive-scaler=4dri4l/predictive-scaler:v3.0-11models"
        
    else
        echo "❌ Push failed!"
    fi
else
    echo "❌ Docker build failed!"
fi