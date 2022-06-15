# ISMLL-Concept-Recommender
Web service recommender that suggests new concepts when students stuck while creating a concept map.

## Prerequisites
The following Python packages are required in order to run the service:

```numpy pandas nltk gensim scipy lightfm scikit-optimize pickle-mixin pathlib flask jsonschema```

In addition:
* `GoogleNews-vectors-negative300` is needed, which can be obtained from [here](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300) (>3GB)
* NLTK requires additional data. The [manual](https://www.nltk.org/data.html) explains how to install this data.


### Issue with lightfm on Windows
Note on Windows `lightfm` requires further the package `wheel` and the Visual Studio Build Tools, which can be obtained from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
Even with this additional software, it's unlikely to install `lightfm` [see here](https://github.com/lyst/lightfm/issues/644) 

## Execution
The algorithm may be executed as Python script for development and testing purposes, as well as web service. The repository contains two scripts to execute the web service on Linux and Windows.

### Local Execution
```python python Webservice.py```

You may edit and execute the following functions:
* `do_training_without_webservice()` for testing the training endpoint

### Web wervice (Linux)
```
chmod +x start.sh
./start.sh
```

### Web Service (Windows)
```start.bat```

## Web Service / Endpoints
The web service runs by default on Port 5000, but may be changed by the first parameter of the start scripts.
The following endpoints are currently provided (Swagger documentation will follow):
* GET: `concept_recommender/hello_world`: Demonstrates that the service is available / how to call a GET endpoint
* POST: `/concept_recommender/train`: Expects an array of concept maps serialized in JSON to train the models.