# coding: utf-8

"""
    Student-Management-System-API

    The Student-Management-System-API. <a href='http://147.172.178.30:8080/stmgmt/api-json'>JSON</a>  # noqa: E501

    OpenAPI spec version: 2.7.2
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class RoundingBehavior(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'type': 'str',
        'decimals': 'float'
    }

    attribute_map = {
        'type': 'type',
        'decimals': 'decimals'
    }

    def __init__(self, type=None, decimals=None):  # noqa: E501
        """RoundingBehavior - a model defined in Swagger"""  # noqa: E501
        self._type = None
        self._decimals = None
        self.discriminator = None
        self.type = type
        if decimals is not None:
            self.decimals = decimals

    @property
    def type(self):
        """Gets the type of this RoundingBehavior.  # noqa: E501


        :return: The type of this RoundingBehavior.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this RoundingBehavior.


        :param type: The type of this RoundingBehavior.  # noqa: E501
        :type: str
        """
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")  # noqa: E501
        allowed_values = ["NONE", "DECIMALS", "DOWN_NEAREST_INTEGER", "UP_NEAREST_INTEGER"]  # noqa: E501
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` ({0}), must be one of {1}"  # noqa: E501
                .format(type, allowed_values)
            )

        self._type = type

    @property
    def decimals(self):
        """Gets the decimals of this RoundingBehavior.  # noqa: E501


        :return: The decimals of this RoundingBehavior.  # noqa: E501
        :rtype: float
        """
        return self._decimals

    @decimals.setter
    def decimals(self, decimals):
        """Sets the decimals of this RoundingBehavior.


        :param decimals: The decimals of this RoundingBehavior.  # noqa: E501
        :type: float
        """

        self._decimals = decimals

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(RoundingBehavior, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, RoundingBehavior):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
