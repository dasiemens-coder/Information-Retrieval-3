# Information Retrieval #3 - LOINC Retrieval

### Authors 
- Davis Siemens
- Inés Simón del Collado
- Xiya Sun

---

## Notebooks & Scripts
The project consists of **two notebook** and several phyton scripts:
1. `LoincRanker.ipynb` – AdaRank Implementation. HTML export of runned notebook is available (LoincRanker.html).
2. `info_retrieval.ipynb` - Notebook used to retrieve training and testing dataset.
3. 'adarankv2.py', 'metrics.py' & 'metrics.py' - Modified version of 2017 Ruey-Cheng Chen AdaRank implementation
---

## Folders

### `/Input`
- Please download '/Input' data from https://1drv.ms/f/s!Av8MbzXuqPyoiJsrxcj7_RmgN9Ww8A?e=tamURT and add it here. 

### `/Results`
- Contains examplary retrievals based on created test dataset. Data can also be accessed over https://1drv.ms/f/s!Av8MbzXuqPyoiJsrxcj7_RmgN9Ww8A?e=tamURT. 

### `/Idxdata`
- Stores retrieved features and indexed data for faster retrieval. 

---

## Execution Instructions
1. Ensure all necessary dependencies are installed.
2. To run the notebook (LoincRanker.ipynb) locally, download the dataset from OneDrive (https://1drv.ms/f/s!Av8MbzXuqPyoiJsrxcj7_RmgN9Ww8A?e=tamURT) and place it in the `/Input` directory.
3. Run 'LoincRanker.ipynb'
4. Alterntative: See results of notebook in LoincRanker.html and results of retrieval in '/Results' without running the notebook. 
