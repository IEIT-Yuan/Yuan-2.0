#!/bin/bash

CONTROLLER_HOST="0.0.0.0"
CONTROLLER_PORT=8503

MODEL_WORKER_HOST="0.0.0.0"
MODEL_WORKER_PORT=8504

API_SERVER_HOST="0.0.0.0"
API_SERVER_PORT=8505

MODEL_PATH="/mnt/models/Yuan2-2B-Mars-hf/"

start_controller() {
    echo "Starting controller service..."
    python3 -m fastchat.serve.controller --host ${CONTROLLER_HOST} --port ${CONTROLLER_PORT} > controller.log 2>&1 &
}

start_model_worker() {
    echo "Starting model worker service..."
    python3 -m fastchat.serve.model_worker --model-path ${MODEL_PATH} --model-names "yuan2" --controller-address http://${CONTROLLER_HOST}:${CONTROLLER_PORT} --worker-address http://${MODEL_WORKER_HOST}:${MODEL_WORKER_PORT} --host ${MODEL_WORKER_HOST} --port ${MODEL_WORKER_PORT} --dtype bfloat16 --debug True > model_worker.log 2>&1 &
}

start_openai_api_server() {
    echo "Starting OpenAI API server..."
    python3 -m fastchat.serve.openai_api_server --host ${API_SERVER_HOST} --port ${API_SERVER_PORT} --controller-address http://${CONTROLLER_HOST}:${CONTROLLER_PORT} > server.log 2>&1 &
}

stop_controller() {
    echo "Stopping controller service..."
    pids=$(pgrep -f "python3 -m fastchat.serve.controller")
    if [ -n "$pids" ]; then
        kill -9 $pids
        echo "Controller service stopped."
    else
        echo "Controller service is not running."
    fi
}

stop_model_worker() {
    echo "Stopping model worker service..."
    pids=$(pgrep -f "python3 -m fastchat.serve.model_worker")
    if [ -n "$pids" ]; then
        kill -9 $pids
        echo "Model worker service stopped."
    else
        echo "Model worker service is not running."
    fi
}

stop_openai_api_server() {
    echo "Stopping OpenAI API server..."
    pids=$(pgrep -f "python3 -m fastchat.serve.openai_api_server")
    if [ -n "$pids" ]; then
        kill -9 $pids
        echo "OpenAI API server stopped."
    else
        echo "OpenAI API server is not running."
    fi
}

stop_services() {
    echo "Stopping services..."
    stop_controller
    stop_model_worker
    stop_openai_api_server
}

check_status() {
    echo "Checking status..."
    if pgrep -f "python3 -m fastchat.serve.controller" &> /dev/null; then
        echo "Controller service is running."
    else
        echo "Controller service is not running."
    fi

    if pgrep -f "python3 -m fastchat.serve.model_worker" &> /dev/null; then
        echo "Model worker service is running."
    else
        echo "Model worker service is not running."
    fi

    if pgrep -f "python3 -m fastchat.serve.openai_api_server" &> /dev/null; then
        echo "OpenAI API server is running."
    else
        echo "OpenAI API server is not running."
    fi
}

case "$1" in
    "start_all")
        start_controller
        sleep 5  # Wait for controller to start before starting other services
        start_model_worker
        sleep 15 # Wait for worker to start before starting other services
        start_openai_api_server
        ;;
    "start_controller")
        start_controller
        ;;
    "start_worker")
        start_model_worker
        ;;
    "start_server")
        start_openai_api_server
        ;;
    "stop_all")
        stop_services
        ;;
    "stop_controller")
        stop_controller
        ;;
    "stop_worker")
        stop_model_worker
        ;;
    "stop_server")
        stop_openai_api_server
        ;;
    "status")
        check_status
        ;;
    *)
        echo "Usage: $0 {start_all|start_controller|start_worker|start_server|stop_all|stop_controller|stop_worker|stop_server|status}"
        exit 1
        ;;
esac

exit 0
