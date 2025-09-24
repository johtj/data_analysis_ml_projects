# Github repository for projects in FYS-STK4155: Applied Data Analysis and Machine Learning
Group consisting of: 
  * Johanna Tjernström
  * Morten Taraldsten Brunes
  * Satu Karoliina Innanen
  * Erik Berthelsen

## Installation and running the code
To install required packages:

Using pip
'''bash
pip install -r requirements.txt
'''
Using conda
'''bash
conda install --yes --file requirements.txt
'''
Once the apropriate environment has been created, and the necessary packages are up to date the main code is run from the central jupyter notebook "assignment*.ipynb", for example using a jupyter environment, or VS code. 

## Directory structure
.
├── assignment1
│   ├── Code
│   │   ├── assigmnet1.ipynb
│   │   ├── matrix_creation.py
│   │   ├── main_methods.py
│   │   ├── errors.py
│   │   ├── plotting.py
│   ├── Report
│   ├── Figures
├── assignment2
│   ├── Code
│   ├── Report
│   ├── Figures
├── assignment3
│   ├── Code
│   ├── Report
│   ├── Figures

Where each assignment has a folder. Within that folder the Code folder contains any code associated with the assignment. The assignment*.ipynb notebook contains the working code for the assignment, and any *.py files contains functionality, or help functions used in the main notebook. The Report folder contains the report, and the Figures folder contains the figures generated in the main notebook, these figures are also used in the report. 

