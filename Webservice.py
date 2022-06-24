# Based on:
# * https://towardsdatascience.com/working-with-apis-using-flask-flask-restplus-and-swagger-ui-7cf447deda7f
# * https://www.imaginarycloud.com/blog/flask-python/#Swagger

from flask import Flask, request
import json, jsonschema
from jsonschema import validate
from flask_restx import Api, Resource, fields
import RecommenderAPI
import inference

flask_app = Flask(__name__)
app = Api(app = flask_app, version='0.1', title='ISMLL Concept Recommender', description='Web service recommender that suggests new concepts when students stuck while creating a concept map.')
ns = app.namespace('Concept Maps Recommender', path='/concept_recommender', description='Main APIs')

prediction_request_schema = {
    'type': 'object',
    'properties': {
        'id': {'type': 'integer'},
        'timestamp': {'type': 'integer'},
        'event': {'type': 'string'},
        'user_id': {'type': 'integer'},
        'conceptmap_id': {'type': 'integer'},
        'conceptmap_name': {'type': ["integer","string"]},
        'conceptmap_tags': {
            'type': 'array',
            'items': {
            },
        },
        'concepts': {
            'type': 'array',
            'items': {
                'type': 'object',
                    'properties': {
                        'id': {'type': 'string'},
                        'name': {'type': 'string'},
                        'timestamp': {'type': 'string'},
                    },
                    'required': ['id', 'name', 'timestamp'],
            },
        },
        'recommender_concept': {'type': 'string'},
    },
    'required': ['id', 'timestamp', 'event', 'user_id', 'conceptmap_id', 'conceptmap_name', 'conceptmap_tags', 'concepts', 'recommender_concept'],
}

training_request_schema = {
    'type': 'array',
    'items': {
        'type': 'object',
        'properties': {
            'id': {'type': 'integer'},
            'timestamp': {'type': 'integer'},
            'event': {'type': 'string'},
            'user_id': {'type': 'integer'},
            'conceptmap_id': {'type': 'integer'},
            'conceptmap_name': {'type': ["integer","string"]},
            'conceptmap_tags': {
                'type': 'array',
                'items': {
                },
            },
            'concepts': {
                'type': 'array',
                'items': {
                    'type': 'object',
                        'properties': {
                            'id': {'type': 'string'},
                            'name': {'type': 'string'},
                            'timestamp': {'type': 'string'},
                        },
                        'required': ['id', 'name', 'timestamp'],
                },
            },
            'recommender_concept': {'type': 'string'},
        },
        'required': ['id', 'timestamp', 'event', 'user_id', 'conceptmap_id', 'conceptmap_name', 'conceptmap_tags', 'concepts', 'recommender_concept'],
    },
    'additionalProperties': False,
    'required': ['list'],
}


training_dto = app.schema_model('Multiple Concept Maps DTO', training_request_schema)
prediction_dto = app.schema_model('Single Concept Maps DTO', prediction_request_schema)

@ns.route('/hello_world')
class Hello(Resource):
    def get(self, **kwargs):
        return RecommenderAPI.hello_world()
      
@ns.route('/training/<string:model_id>')
class Training(Resource):
    @ns.expect(training_dto, validate=True)
    def post(self, model_id):
        if request.is_json:
            data = request.get_json()
            
            RecommenderAPI.do_training(data, model_id)
            return "Model updated", 201 # 201 means something was created
        return {"Error": "Request must be JSON"}, 415 # 415 means Unsupported media type
        
@ns.route('/prediction/<string:model_id>')
class Prediction(Resource):
    @ns.expect(prediction_dto, validate=True)
    def get(self, model_id):
        if request.is_json:
            data = request.get_json()
            
            return inference.do_prediction(data, model_id), 200
        return {"Error": "Request must be JSON"}, 415 # 415 means Unsupported media type