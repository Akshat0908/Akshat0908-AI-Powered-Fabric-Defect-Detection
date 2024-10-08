# AI-Powered Fabric Defect Detection

An AI-driven solution for automated fabric defect detection using deep learning and computer vision.

<img width="848" alt="Screenshot 2024-10-02 at 5 59 56 PM" src="https://github.com/user-attachments/assets/923a5625-ded5-442a-9e1d-0e1816c795f1">



## Features
- Upload and analyze fabric images
  
  <img width="1226" alt="Screenshot 2024-10-02 at 5 45 54 PM" src="https://github.com/user-attachments/assets/c6b5fa42-9be6-4300-9b53-bbff01a4b44e">


- Defect detection using MobileNetV2

  <img width="843" alt="Screenshot 2024-10-02 at 5 46 05 PM" src="https://github.com/user-attachments/assets/6219721d-1ccf-49a2-b7e4-5ef3e343626e">

- Grad-CAM heatmap visualization
- Contextual analysis and recommendations

  <img width="1081" alt="Screenshot 2024-10-02 at 5 46 11 PM" src="https://github.com/user-attachments/assets/d430f8d7-378c-4a9b-a997-3356ae901a0f">


## Usage
1. Run the Streamlit app:
   ```
   streamlit run fabric_defect_detection.py
   ```

2. Access the web interface via your browser (typically http://localhost:8501)

3. Upload a fabric image using the file uploader

4. View and interpret the results:
   - Original image
   - Defect heatmap
   - Detection result and confidence score
   - Explanation of the heatmap
   - Contextual analysis of potential defect causes
   - Recommendations for addressing the detected issues
  
  

## Limitations and Future Enhancements
- Currently limited to binary classification (defect/no defect)
- Future plans include:
  - Multi-class defect classification
  - Custom model training on textile-specific datasets
  - Real-time analysis capabilities for production lines
  - Integration with manufacturing systems and IoT devices



## Technologies
Python, TensorFlow, OpenCV, Streamlit, NumPy, Matplotlib

## Future Enhancements
- Multi-class defect classification
- Custom dataset training
- Real-time analysis
- Manufacturing system integration
