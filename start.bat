set PORT=5000

echo "Starting ISMLL Recommener at %PORT%"
echo "See for documentation: http://localhost:%PORT%/"


set FLASK_APP=Webservice.py
set FLASK_ENV=development
flask run --host=0.0.0.0 --port=%PORT%