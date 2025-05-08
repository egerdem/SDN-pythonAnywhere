import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your manager and visualizer
from sdn_manager_load_sims_pythonANY import ExperimentLoaderManager

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, './results')

# Initialize the manager with your specific project
manager = ExperimentLoaderManager(
    results_dir=RESULTS_DIR,
    is_batch_manager=True,  # Adjust based on your data structure
    project_names=["aes_SINGLE", "journal_SINGLE", "aes_MULTI"],  # Adjust to your specific folder
    project_source_filters={"aes_MULTI": ["Center_Source"]} # Only load Center_Source for aes_MULTI
)

# Diagnostic prints for PythonAnywhere
print(f"PYTHONANYWHERE_FLASK_APP: RESULTS_DIR = {RESULTS_DIR}")
try:
    print(f"PYTHONANYWHERE_FLASK_APP: singular_manager.projects = {manager.projects}")
    if not manager.projects:
        print("PYTHONANYWHERE_FLASK_APP: singular_manager.projects is EMPTY.")
    else:
        print("PYTHONANYWHERE_FLASK_APP: singular_manager.projects is POPULATED.")
except AttributeError:
    print("PYTHONANYWHERE_FLASK_APP: singular_manager has no attribute 'projects'.")
except Exception as e:
    print(f"PYTHONANYWHERE_FLASK_APP: Error accessing singular_manager.projects: {e}")

from sdn_experiment_visualizer_PYTHONANYWHERE import ExperimentVisualizer
# Create the visualizer
visualizer = ExperimentVisualizer(manager)

# Configure the app for deployment
app = visualizer.create_app_for_deployment()
