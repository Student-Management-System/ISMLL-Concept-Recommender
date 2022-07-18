# Based on:
# * https://towardsdatascience.com/working-with-apis-using-flask-flask-restplus-and-swagger-ui-7cf447deda7f
# * https://www.imaginarycloud.com/blog/flask-python/#Swagger

from flask import Flask, request
import json, jsonschema
from jsonschema import validate
from flask_restx import Api, Resource, fields
import concept_map_recommender_training_api
import concept_map_recommender_inference_api
import group_recommender_batch_api

flask_app = Flask(__name__)
app = Api(app = flask_app, version='0.2', title='ISMLL Quality+ Recommenders', description='Web service recommenders of the Quality+ project. Concept Map Recommender suggests new concepts when students stuck while creating a concept map. Group recommender suggest new partners for groups that do not fullfill the group requirements.')

cmr = app.namespace('Concept Maps Recommender', path='/concept_recommender', description='APIs for the Concept Map Recommender')

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

@cmr.route('/hello_world')
class Hello(Resource):
    def get(self, **kwargs):
        return concept_map_recommender_training_api.hello_world()
      
@cmr.route('/training/<string:model_id>')
class Concept_Map_Training(Resource):
    @cmr.expect(training_dto, validate=True)
    def post(self, model_id):
        if request.is_json:
            data = request.get_json()
            
            concept_map_recommender_training_api.do_training(data, model_id)
            return "Model updated", 201 # 201 means something was created
        return {"Error": "Request must be JSON"}, 415 # 415 means Unsupported media type
        
@cmr.route('/prediction/<string:model_id>')
class Concept_Map_Prediction(Resource):
    @cmr.expect(prediction_dto, validate=True)
    def post(self, model_id):
        if request.is_json:
            data = request.get_json()
            
            return concept_map_recommender_inference_api.do_prediction(data, model_id), 200
        return {"Error": "Request must be JSON"}, 415 # 415 means Unsupported media type
   
   
gr  = app.namespace('Group Recommender', path='/group_recommender', description='APIs for the Group Recommender')

group_recommendation_request_schema = {
    'type': 'object',
    'properties': {
        'course': {'type': 'object'},
    },
    'required': ['course'],
}

group_dto = app.schema_model('Group Recommendation DTO', group_recommendation_request_schema)
  
@gr.route('/nn')
class Group_NN_Prediction(Resource):
    @gr.expect(group_dto, validate=False)
    def post(self):
        if request.is_json:
            data = request.get_json()
            
            return group_recommender_batch_api.group_recommendation_with_neuronal_network(data), 200
        return {"Error": "Request must be JSON"}, 415 # 415 means Unsupported media type