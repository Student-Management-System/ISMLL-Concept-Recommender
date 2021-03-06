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

class ParticipantsComparisonDto(object):
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
        'in_compared_courses': 'list[UserDto]',
        'not_in_compared_courses': 'list[UserDto]'
    }

    attribute_map = {
        'in_compared_courses': 'inComparedCourses',
        'not_in_compared_courses': 'notInComparedCourses'
    }

    def __init__(self, in_compared_courses=None, not_in_compared_courses=None):  # noqa: E501
        """ParticipantsComparisonDto - a model defined in Swagger"""  # noqa: E501
        self._in_compared_courses = None
        self._not_in_compared_courses = None
        self.discriminator = None
        self.in_compared_courses = in_compared_courses
        self.not_in_compared_courses = not_in_compared_courses

    @property
    def in_compared_courses(self):
        """Gets the in_compared_courses of this ParticipantsComparisonDto.  # noqa: E501


        :return: The in_compared_courses of this ParticipantsComparisonDto.  # noqa: E501
        :rtype: list[UserDto]
        """
        return self._in_compared_courses

    @in_compared_courses.setter
    def in_compared_courses(self, in_compared_courses):
        """Sets the in_compared_courses of this ParticipantsComparisonDto.


        :param in_compared_courses: The in_compared_courses of this ParticipantsComparisonDto.  # noqa: E501
        :type: list[UserDto]
        """
        if in_compared_courses is None:
            raise ValueError("Invalid value for `in_compared_courses`, must not be `None`")  # noqa: E501

        self._in_compared_courses = in_compared_courses

    @property
    def not_in_compared_courses(self):
        """Gets the not_in_compared_courses of this ParticipantsComparisonDto.  # noqa: E501


        :return: The not_in_compared_courses of this ParticipantsComparisonDto.  # noqa: E501
        :rtype: list[UserDto]
        """
        return self._not_in_compared_courses

    @not_in_compared_courses.setter
    def not_in_compared_courses(self, not_in_compared_courses):
        """Sets the not_in_compared_courses of this ParticipantsComparisonDto.


        :param not_in_compared_courses: The not_in_compared_courses of this ParticipantsComparisonDto.  # noqa: E501
        :type: list[UserDto]
        """
        if not_in_compared_courses is None:
            raise ValueError("Invalid value for `not_in_compared_courses`, must not be `None`")  # noqa: E501

        self._not_in_compared_courses = not_in_compared_courses

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
        if issubclass(ParticipantsComparisonDto, dict):
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
        if not isinstance(other, ParticipantsComparisonDto):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
