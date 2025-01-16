# TheRealEye: Deepfake Detection Application

## Overview

TheRealEye is a web-based application designed to detect deepfake videos. Users can upload videos or images, and the application will analyze them using a light-weight but powerful deep learning model created by me, achieving 90% accuracy to determine whether the content is real or fake.

---

## Project Structure

```
TheRealEye/
|- models/
|  |- Deep-Fake_Detection_FINALMODEL.pkl
|  |- Demo_TheRealEyeModel1.pkl
|- static/
|  |- (Assets like CSS, JS, Images)
|- templates/
|  |- (HTML Templates Here)
|- therealeye/
|  |- Include/
|  |- Lib/
|  |- Scripts/
|  |- pyvenv.cfg
|- uploads/
|- app.py          # Main Application
|- asd.py          # Experimental Script
|- modification.py # Modification Logic
|- README.md       # Project Guide
|- requirements.txt # Dependencies
|- true_modified_file.csv # Data File
```

---

## Features

- Upload videos or images.
- Detect deepfakes using pre-trained EfficientNet models.
- Display results with accuracy metrics.

---

## Requirements

Ensure you have the following installed:

- Python 3.8+
- Flask
- OpenCV
- PyTorch
- EfficientNet-PyTorch

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Setup and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Kaustav-coder-hub/DeepFakeDetection-model
// This command navigates to the repository directory. 
// Note: The name of the repository directory may differ on your PC.
cd TheRealEye
```

### 2. Setup Virtual Environment

```bash
python -m venv therealeye
therealeye/Scripts/activate  # On Windows
source therealeye/bin/activate      # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Access the application in your browser at `http://127.0.0.1:5000/`.

---

## Folder Explanations

- **models/**: Contains pre-trained models for deepfake detection.
- **static/**: Stores CSS, JavaScript, and other static files.
- **templates/**: Holds HTML templates for the web interface.
- **uploads/**: Temporary folder for user-uploaded files.
- **app.py**: Main script to run the application.
- **requirements.txt**: List of all Python dependencies.

---

## Troubleshooting

1. **ModuleNotFoundError**: Ensure all dependencies are installed with `pip install -r requirements.txt`.
2. **Memory Errors**: Switch to a GPU-enabled environment if possible.
3. **Server Issues**: Check if Flask is running on the correct port (default: 5000).

---

## Future Enhancements

- Add support for real-time deepfake detection.
- Implement user authentication and video history tracking.
- Improve model performance with additional training.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributions

Feel free to fork this repository, submit pull requests, or report issues. Contributions are welcome!

---

## Contact

For questions or support, contact [[therealeye4@gmail.com](mailto\:therealeye4@gmail.com)].

