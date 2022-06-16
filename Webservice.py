# Based on:
# * https://towardsdatascience.com/working-with-apis-using-flask-flask-restplus-and-swagger-ui-7cf447deda7f
# * https://www.imaginarycloud.com/blog/flask-python/#Swagger

from flask import Flask, request
import json, jsonschema
from jsonschema import validate
from flask_restx import Api, Resource, fields
import RecommenderAPI

flask_app = Flask(__name__)
app = Api(app = flask_app, version='0.1', title='ISMLL Concept Recommender', description='Web service recommender that suggests new concepts when students stuck while creating a concept map.')
ns = app.namespace('concept_recommender', description='Main APIs')

request_schema = {
    'type': 'object',
    'properties': {
        'list': {
            'type': 'array',
            'items': {'type': 'string'},
            'minItems': 1,
        },
    },
    'additionalProperties': False,
    'required': ['list'],
}
request_model = app.schema_model('test_list_request', request_schema)

@ns.route('/hello_world')
class Hello(Resource):
    def get(self, **kwargs):
        return RecommenderAPI.hello_world()
      
@ns.route('/training/<string:model_id>')
class Training(Resource):
    @ns.expect(request_model, validate=True)
    def post(self, todo_id):
        if request.is_json:
            data = request.get_json()
            valid, msg = RecommenderAPI.validate_json(data, 'Multiple-Concept-Maps-Recommender.spec.json')

            if not valid:
                return {"Error": "JSON request does not represent Concept Map(s):\n" + msg}, 415 # 415 means Unsupported media type

            RecommenderAPI.do_training(data, todo_id)
            return "Model updated", 201 # 201 means something was created
        return {"Error": "Request must be JSON"}, 415 # 415 means Unsupported media type