# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter client."""

import logging

import grpc

from pb import learn_pb2, learn_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = learn_pb2_grpc.LearningCenterStub(channel)

        response = stub.StartTrain(learn_pb2.TrainCommand(order=1, data_from_near_day=7, limit_hour=3, target_accuracy=1.00))
        print("client received StartTrain ETA: " + response.ETA)
        
        for i in range(10):
            response = stub.Touch(learn_pb2.BasicRequest(str='test touch', size=i))
            print("client received Touch: " + response.message)
        
        


if __name__ == '__main__':
    logging.basicConfig()
    run()
