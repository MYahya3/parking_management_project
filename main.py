# controller.py
import time
from flask import Flask, jsonify
from detection import run_detection, load_json
import threading
from utilis import load_json

# Flask app setup for the API trigger
app = Flask(__name__)


# Time interval in seconds (e.g., 20 seconds)
json_data = load_json("parameters.json")
timer_to_get_result = json_data["schedule_to_run_detection"]


# Global flag to trigger frame processing via API
api_triggered = False

# Last time detection was run
last_detection_time = time.time()

@app.route('/trigger-detection', methods=['POST'])
def process_frame_trigger():
    global api_triggered
    api_triggered = True  # Set flag to true to process frame
    return jsonify({"status": "success", "message": "Frame processing triggered"}), 200

def start_flask_app():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # Start the Flask API server in a separate thread
    flask_thread = threading.Thread(target=start_flask_app, daemon=True)
    flask_thread.start()

    try:
        while True:
            # Get the current time
            current_time = time.time()

            # Check if time interval has passed or API was triggered
            if (current_time - last_detection_time >= timer_to_get_result) or api_triggered:
                # Run the detection function
                print("Running detection...")
                run_detection(json_file = "parameters.json")

                # Reset the last detection time and API trigger flag
                last_detection_time = current_time
                api_triggered = False

    except KeyboardInterrupt:
        print("Shutting down...")
