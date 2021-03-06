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

class CourseCreateDto(object):
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
        'id': 'str',
        'shortname': 'str',
        'semester': 'str',
        'title': 'str',
        'is_closed': 'bool',
        'config': 'CourseConfigDto',
        'lecturers': 'list[str]',
        'links': 'list[LinkDto]'
    }

    attribute_map = {
        'id': 'id',
        'shortname': 'shortname',
        'semester': 'semester',
        'title': 'title',
        'is_closed': 'isClosed',
        'config': 'config',
        'lecturers': 'lecturers',
        'links': 'links'
    }

    def __init__(self, id=None, shortname=None, semester=None, title=None, is_closed=None, config=None, lecturers=None, links=None):  # noqa: E501
        """CourseCreateDto - a model defined in Swagger"""  # noqa: E501
        self._id = None
        self._shortname = None
        self._semester = None
        self._title = None
        self._is_closed = None
        self._config = None
        self._lecturers = None
        self._links = None
        self.discriminator = None
        self.id = id
        self.shortname = shortname
        self.semester = semester
        self.title = title
        self.is_closed = is_closed
        self.config = config
        if lecturers is not None:
            self.lecturers = lecturers
        if links is not None:
            self.links = links

    @property
    def id(self):
        """Gets the id of this CourseCreateDto.  # noqa: E501

        Unique identifier of this course.  # noqa: E501

        :return: The id of this CourseCreateDto.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this CourseCreateDto.

        Unique identifier of this course.  # noqa: E501

        :param id: The id of this CourseCreateDto.  # noqa: E501
        :type: str
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def shortname(self):
        """Gets the shortname of this CourseCreateDto.  # noqa: E501

        Shortname of this course, i.e 'java'. Should be reused every semester. Will be used in URLs.  # noqa: E501

        :return: The shortname of this CourseCreateDto.  # noqa: E501
        :rtype: str
        """
        return self._shortname

    @shortname.setter
    def shortname(self, shortname):
        """Sets the shortname of this CourseCreateDto.

        Shortname of this course, i.e 'java'. Should be reused every semester. Will be used in URLs.  # noqa: E501

        :param shortname: The shortname of this CourseCreateDto.  # noqa: E501
        :type: str
        """
        if shortname is None:
            raise ValueError("Invalid value for `shortname`, must not be `None`")  # noqa: E501

        self._shortname = shortname

    @property
    def semester(self):
        """Gets the semester of this CourseCreateDto.  # noqa: E501

        Semester that the iteration of this course belong to.  # noqa: E501

        :return: The semester of this CourseCreateDto.  # noqa: E501
        :rtype: str
        """
        return self._semester

    @semester.setter
    def semester(self, semester):
        """Sets the semester of this CourseCreateDto.

        Semester that the iteration of this course belong to.  # noqa: E501

        :param semester: The semester of this CourseCreateDto.  # noqa: E501
        :type: str
        """
        if semester is None:
            raise ValueError("Invalid value for `semester`, must not be `None`")  # noqa: E501

        self._semester = semester

    @property
    def title(self):
        """Gets the title of this CourseCreateDto.  # noqa: E501

        The full title of this course, i.e Programming I: Java  # noqa: E501

        :return: The title of this CourseCreateDto.  # noqa: E501
        :rtype: str
        """
        return self._title

    @title.setter
    def title(self, title):
        """Sets the title of this CourseCreateDto.

        The full title of this course, i.e Programming I: Java  # noqa: E501

        :param title: The title of this CourseCreateDto.  # noqa: E501
        :type: str
        """
        if title is None:
            raise ValueError("Invalid value for `title`, must not be `None`")  # noqa: E501

        self._title = title

    @property
    def is_closed(self):
        """Gets the is_closed of this CourseCreateDto.  # noqa: E501

        Determines, wether changes (i.e joining this course) can be made to this course.  # noqa: E501

        :return: The is_closed of this CourseCreateDto.  # noqa: E501
        :rtype: bool
        """
        return self._is_closed

    @is_closed.setter
    def is_closed(self, is_closed):
        """Sets the is_closed of this CourseCreateDto.

        Determines, wether changes (i.e joining this course) can be made to this course.  # noqa: E501

        :param is_closed: The is_closed of this CourseCreateDto.  # noqa: E501
        :type: bool
        """
        if is_closed is None:
            raise ValueError("Invalid value for `is_closed`, must not be `None`")  # noqa: E501

        self._is_closed = is_closed

    @property
    def config(self):
        """Gets the config of this CourseCreateDto.  # noqa: E501


        :return: The config of this CourseCreateDto.  # noqa: E501
        :rtype: CourseConfigDto
        """
        return self._config

    @config.setter
    def config(self, config):
        """Sets the config of this CourseCreateDto.


        :param config: The config of this CourseCreateDto.  # noqa: E501
        :type: CourseConfigDto
        """
        if config is None:
            raise ValueError("Invalid value for `config`, must not be `None`")  # noqa: E501

        self._config = config

    @property
    def lecturers(self):
        """Gets the lecturers of this CourseCreateDto.  # noqa: E501


        :return: The lecturers of this CourseCreateDto.  # noqa: E501
        :rtype: list[str]
        """
        return self._lecturers

    @lecturers.setter
    def lecturers(self, lecturers):
        """Sets the lecturers of this CourseCreateDto.


        :param lecturers: The lecturers of this CourseCreateDto.  # noqa: E501
        :type: list[str]
        """

        self._lecturers = lecturers

    @property
    def links(self):
        """Gets the links of this CourseCreateDto.  # noqa: E501


        :return: The links of this CourseCreateDto.  # noqa: E501
        :rtype: list[LinkDto]
        """
        return self._links

    @links.setter
    def links(self, links):
        """Sets the links of this CourseCreateDto.


        :param links: The links of this CourseCreateDto.  # noqa: E501
        :type: list[LinkDto]
        """

        self._links = links

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
        if issubclass(CourseCreateDto, dict):
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
        if not isinstance(other, CourseCreateDto):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
