1. Create a python env 
windows: 
pip install virtualenv 
python -m venv myenv 
myenv\Scripts\activate 

2. Install required libraries 
pip install transformers pillow psutil torch pyttsx3 pandas pytesseract scikit
learn torchvision nltk rouge-score googletrans==4.0.0-rc1 
pip install -r requirements.txt 

3. Install Tesseract-OCR (System Dependency) 
On Windows 
Download the Tesseract installer from Tesseract GitHub Releases. 
https://github.com/UB-Mannheim/tesseract/wiki  
Install it and note the installation path (e.g., C:\Program Files\Tesseract
OCR\tesseract.exe). 
Manually Set System Environment Variable 
Open System Properties: 
Press Win + R, type sysdm.cpl, and press Enter. 
        Go to the Advanced tab and click Environment Variables. 
        Add or Edit a Variable: 
 
        In the System variables section: 
        To add a new variable, click New. 
        To edit an existing variable (e.g., Path), select it and click Edit. 
        Set the Variable: 
 
        For Tesseract, add the following path to the Path variable 
  
         C:\Program Files\Tesseract-OCR 
        Save Changes: 
 
        Click OK to save and close all dialogs. 
        Verify Tesseract Installation(Windows) 
        Run the following command to ensure Tesseract is properly installed: 
 
         tesseract --version 
  . 
 
 
4. Run file "(whichever file the GUI has)"  
Run the following command in cmd  
python fine_tuned.py 
Alternatively you can try running it by opening the fine_tuned.exe file in the dist 
folder of the program. 
Image-Captioning-Program\dist\fine_tune.exe