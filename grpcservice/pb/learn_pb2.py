# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: learn.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='learn.proto',
  package='',
  syntax='proto3',
  serialized_options=b'\242\002\002LC',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0blearn.proto\"6\n\x0c\x42\x61sicRequest\x12\x0c\n\x04size\x18\x01 \x01(\x03\x12\x0b\n\x03str\x18\x02 \x01(\t\x12\x0b\n\x03num\x18\x03 \x01(\x05\"\x1d\n\nBasicReply\x12\x0f\n\x07message\x18\x01 \x01(\t\"\x1d\n\x0c\x42\x61sicCommand\x12\r\n\x05order\x18\x01 \x01(\x05\"f\n\x0cTrainCommand\x12\r\n\x05order\x18\x01 \x01(\x05\x12\x1a\n\x12\x64\x61ta_from_near_day\x18\x02 \x01(\x05\x12\x12\n\nlimit_hour\x18\x03 \x01(\x05\x12\x17\n\x0ftarget_accuracy\x18\x04 \x01(\x01\"W\n\tTrainInfo\x12\x0e\n\x06thread\x18\x01 \x01(\x05\x12\r\n\x05total\x18\x02 \x01(\x05\x12\x0c\n\x04loss\x18\x03 \x01(\x01\x12\x10\n\x08\x61\x63\x63uracy\x18\x04 \x01(\x01\x12\x0b\n\x03\x45TA\x18\x05 \x01(\t2\xfd\x01\n\x0eLearningCenter\x12%\n\x05Touch\x12\r.BasicRequest\x1a\x0b.BasicReply\"\x00\x12)\n\nStartTrain\x12\r.TrainCommand\x1a\n.TrainInfo\"\x00\x12(\n\tStopTrain\x12\r.BasicCommand\x1a\n.TrainInfo\"\x00\x12\x35\n\x16GetTrainProcessingInfo\x12\r.BasicCommand\x1a\n.TrainInfo\"\x00\x12\x38\n\x15StreamTrainProcessing\x12\r.BasicCommand\x1a\n.TrainInfo\"\x00(\x01\x30\x01\x42\x05\xa2\x02\x02LCb\x06proto3'
)




_BASICREQUEST = _descriptor.Descriptor(
  name='BasicRequest',
  full_name='BasicRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='size', full_name='BasicRequest.size', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='str', full_name='BasicRequest.str', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num', full_name='BasicRequest.num', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15,
  serialized_end=69,
)


_BASICREPLY = _descriptor.Descriptor(
  name='BasicReply',
  full_name='BasicReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='BasicReply.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=71,
  serialized_end=100,
)


_BASICCOMMAND = _descriptor.Descriptor(
  name='BasicCommand',
  full_name='BasicCommand',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='order', full_name='BasicCommand.order', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=102,
  serialized_end=131,
)


_TRAINCOMMAND = _descriptor.Descriptor(
  name='TrainCommand',
  full_name='TrainCommand',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='order', full_name='TrainCommand.order', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_from_near_day', full_name='TrainCommand.data_from_near_day', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='limit_hour', full_name='TrainCommand.limit_hour', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='target_accuracy', full_name='TrainCommand.target_accuracy', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=133,
  serialized_end=235,
)


_TRAININFO = _descriptor.Descriptor(
  name='TrainInfo',
  full_name='TrainInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='thread', full_name='TrainInfo.thread', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total', full_name='TrainInfo.total', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='loss', full_name='TrainInfo.loss', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='accuracy', full_name='TrainInfo.accuracy', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ETA', full_name='TrainInfo.ETA', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=237,
  serialized_end=324,
)

DESCRIPTOR.message_types_by_name['BasicRequest'] = _BASICREQUEST
DESCRIPTOR.message_types_by_name['BasicReply'] = _BASICREPLY
DESCRIPTOR.message_types_by_name['BasicCommand'] = _BASICCOMMAND
DESCRIPTOR.message_types_by_name['TrainCommand'] = _TRAINCOMMAND
DESCRIPTOR.message_types_by_name['TrainInfo'] = _TRAININFO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BasicRequest = _reflection.GeneratedProtocolMessageType('BasicRequest', (_message.Message,), {
  'DESCRIPTOR' : _BASICREQUEST,
  '__module__' : 'learn_pb2'
  # @@protoc_insertion_point(class_scope:BasicRequest)
  })
_sym_db.RegisterMessage(BasicRequest)

BasicReply = _reflection.GeneratedProtocolMessageType('BasicReply', (_message.Message,), {
  'DESCRIPTOR' : _BASICREPLY,
  '__module__' : 'learn_pb2'
  # @@protoc_insertion_point(class_scope:BasicReply)
  })
_sym_db.RegisterMessage(BasicReply)

BasicCommand = _reflection.GeneratedProtocolMessageType('BasicCommand', (_message.Message,), {
  'DESCRIPTOR' : _BASICCOMMAND,
  '__module__' : 'learn_pb2'
  # @@protoc_insertion_point(class_scope:BasicCommand)
  })
_sym_db.RegisterMessage(BasicCommand)

TrainCommand = _reflection.GeneratedProtocolMessageType('TrainCommand', (_message.Message,), {
  'DESCRIPTOR' : _TRAINCOMMAND,
  '__module__' : 'learn_pb2'
  # @@protoc_insertion_point(class_scope:TrainCommand)
  })
_sym_db.RegisterMessage(TrainCommand)

TrainInfo = _reflection.GeneratedProtocolMessageType('TrainInfo', (_message.Message,), {
  'DESCRIPTOR' : _TRAININFO,
  '__module__' : 'learn_pb2'
  # @@protoc_insertion_point(class_scope:TrainInfo)
  })
_sym_db.RegisterMessage(TrainInfo)


DESCRIPTOR._options = None

_LEARNINGCENTER = _descriptor.ServiceDescriptor(
  name='LearningCenter',
  full_name='LearningCenter',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=327,
  serialized_end=580,
  methods=[
  _descriptor.MethodDescriptor(
    name='Touch',
    full_name='LearningCenter.Touch',
    index=0,
    containing_service=None,
    input_type=_BASICREQUEST,
    output_type=_BASICREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StartTrain',
    full_name='LearningCenter.StartTrain',
    index=1,
    containing_service=None,
    input_type=_TRAINCOMMAND,
    output_type=_TRAININFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StopTrain',
    full_name='LearningCenter.StopTrain',
    index=2,
    containing_service=None,
    input_type=_BASICCOMMAND,
    output_type=_TRAININFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetTrainProcessingInfo',
    full_name='LearningCenter.GetTrainProcessingInfo',
    index=3,
    containing_service=None,
    input_type=_BASICCOMMAND,
    output_type=_TRAININFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StreamTrainProcessing',
    full_name='LearningCenter.StreamTrainProcessing',
    index=4,
    containing_service=None,
    input_type=_BASICCOMMAND,
    output_type=_TRAININFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_LEARNINGCENTER)

DESCRIPTOR.services_by_name['LearningCenter'] = _LEARNINGCENTER

# @@protoc_insertion_point(module_scope)
