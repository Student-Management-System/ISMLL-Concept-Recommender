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

class GroupDto(object):
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
        'name': 'str',
        'password': 'str',
        'has_password': 'bool',
        'size': 'float',
        'is_closed': 'bool',
        'members': 'list[ParticipantDto]',
        'history': 'list[GroupEventDto]'
    }

    attribute_map = {
        'id': 'id',
        'name': 'name',
        'password': 'password',
        'has_password': 'hasPassword',
        'size': 'size',
        'is_closed': 'isClosed',
        'members': 'members',
        'history': 'history'
    }

    def __init__(self, id=None, name=None, password=None, has_password=None, size=None, is_closed=None, members=None, history=None):  # noqa: E501
        """GroupDto - a model defined in Swagger"""  # noqa: E501
        self._id = None
        self._name = None
        self._password = None
        self._has_password = None
        self._size = None
        self._is_closed = None
        self._members = None
        self._history = None
        self.discriminator = None
        self.id = id
        self.name = name
        if password is not None:
            self.password = password
        if has_password is not None:
            self.has_password = has_password
        if size is not None:
            self.size = size
        if is_closed is not None:
            self.is_closed = is_closed
        if members is not None:
            self.members = members
        if history is not None:
            self.history = history

    @property
    def id(self):
        """Gets the id of this GroupDto.  # noqa: E501

        Unique identifier of this group.  # noqa: E501

        :return: The id of this GroupDto.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this GroupDto.

        Unique identifier of this group.  # noqa: E501

        :param id: The id of this GroupDto.  # noqa: E501
        :type: str
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def name(self):
        """Gets the name of this GroupDto.  # noqa: E501

        Name of the group.  # noqa: E501

        :return: The name of this GroupDto.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this GroupDto.

        Name of the group.  # noqa: E501

        :param name: The name of this GroupDto.  # noqa: E501
        :type: str
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

    @property
    def password(self):
        """Gets the password of this GroupDto.  # noqa: E501

        Password required to enter the group.  # noqa: E501

        :return: The password of this GroupDto.  # noqa: E501
        :rtype: str
        """
        return self._password

    @password.setter
    def password(self, password):
        """Sets the password of this GroupDto.

        Password required to enter the group.  # noqa: E501

        :param password: The password of this GroupDto.  # noqa: E501
        :type: str
        """

        self._password = password

    @property
    def has_password(self):
        """Gets the has_password of this GroupDto.  # noqa: E501

        Indicates, wether group has a password. Set by the server.  # noqa: E501

        :return: The has_password of this GroupDto.  # noqa: E501
        :rtype: bool
        """
        return self._has_password

    @has_password.setter
    def has_password(self, has_password):
        """Sets the has_password of this GroupDto.

        Indicates, wether group has a password. Set by the server.  # noqa: E501

        :param has_password: The has_password of this GroupDto.  # noqa: E501
        :type: bool
        """

        self._has_password = has_password

    @property
    def size(self):
        """Gets the size of this GroupDto.  # noqa: E501

        Count of group members. Set by the server.  # noqa: E501

        :return: The size of this GroupDto.  # noqa: E501
        :rtype: float
        """
        return self._size

    @size.setter
    def size(self, size):
        """Sets the size of this GroupDto.

        Count of group members. Set by the server.  # noqa: E501

        :param size: The size of this GroupDto.  # noqa: E501
        :type: float
        """

        self._size = size

    @property
    def is_closed(self):
        """Gets the is_closed of this GroupDto.  # noqa: E501

        Determines, wether course participant are able to join this group.  # noqa: E501

        :return: The is_closed of this GroupDto.  # noqa: E501
        :rtype: bool
        """
        return self._is_closed

    @is_closed.setter
    def is_closed(self, is_closed):
        """Sets the is_closed of this GroupDto.

        Determines, wether course participant are able to join this group.  # noqa: E501

        :param is_closed: The is_closed of this GroupDto.  # noqa: E501
        :type: bool
        """

        self._is_closed = is_closed

    @property
    def members(self):
        """Gets the members of this GroupDto.  # noqa: E501


        :return: The members of this GroupDto.  # noqa: E501
        :rtype: list[ParticipantDto]
        """
        return self._members

    @members.setter
    def members(self, members):
        """Sets the members of this GroupDto.


        :param members: The members of this GroupDto.  # noqa: E501
        :type: list[ParticipantDto]
        """

        self._members = members

    @property
    def history(self):
        """Gets the history of this GroupDto.  # noqa: E501


        :return: The history of this GroupDto.  # noqa: E501
        :rtype: list[GroupEventDto]
        """
        return self._history

    @history.setter
    def history(self, history):
        """Sets the history of this GroupDto.


        :param history: The history of this GroupDto.  # noqa: E501
        :type: list[GroupEventDto]
        """

        self._history = history

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
        if issubclass(GroupDto, dict):
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
        if not isinstance(other, GroupDto):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
