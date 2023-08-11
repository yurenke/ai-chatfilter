from django.apps import AppConfig
from concurrent import futures
import logging
import grpc, time, random
from grpcservice.pb import learn_pb2_grpc, learn_pb2

from ai.train import train_pinyin_to_next_version
from service import instance

interal_pinyin_model = None


class GrpcService():
    """
    """
    host = ''
    port = 0
    webhost = ''
    webport = 80
    server = None
    service_instance = None
    executor = None

    def __init__(self, host, port, webhost, webport = 80):
        self.host = host
        self.port = port
        self.webhost = webhost
        self.webport = webport
        self.service_instance = instance.get_main_service(is_admin=False)
        self.launch()
        

    def launch(self):
        self.service_instance.fetch_ai_model_data(remote_ip=self.webhost, port=self.webport)
        self.executor = futures.ThreadPoolExecutor(max_workers=8)
        self.server = grpc.server(self.executor)
        learn_pb2_grpc.add_LearningCenterServicer_to_server(LearningServicer(service_instance=self.service_instance), self.server)

        self.server.add_insecure_port('[::]:{}'.format(self.port))
        self.server.start()
        self.server.wait_for_termination()



class LearningServicer(learn_pb2_grpc.LearningCenterServicer):
    """
    """
    ORDER_PINYIN = 1
    times = 0
    service_instance = None
    learn_handler = None
    
    def __init__(self, service_instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_instance = service_instance
        self.learn_handler = LearningHandler()
        print('Will LearningServicer Init???')


    def Touch(self, request, context):
        
        self.times += 1
        message = 'size: {} str: {} num: {} times: {}'.format(request.size, request.str, request.num, self.times)
        logging.debug('request:  {}'.format(message))
        print('Touch: ', self.service_instance.vocabulary_data[:5])
        return learn_pb2.BasicReply(message=message)

    
    def StartTrain(self, request, context):
        _order = request.order
        if _order == self.ORDER_PINYIN:
            logging.debug('ORDER_PINYIN')
        else:
            pass
        # request.data_from_near_day
        # request.limit_hour
        # request.target_accuracy
        # self.trainPinyin
        time.sleep(30 + (round(random.random() * 10)))
        return learn_pb2.TrainInfo(thread=1, total=10, loss=0.99, accuracy=0.99, ETA='ORDER_PINYIN')


    def StopTrain(self, request, context):
        return learn_pb2.TrainInfo()


    def GetTrainProcessingInfo(self, request, context):
        return learn_pb2.TrainInfo()


    def StreamTrainProcessing(self, request, context):
        return learn_pb2.TrainInfo()





class LearningHandler():
    """
    """
    executor = None

    def __init__(self):
        self.executor = futures.ProcessPoolExecutor(max_workers=2)


    def trainPinyin(self, dataset_list, max_spend_time):
        jieba_vocabulary = []
        jieba_freqs = []
        # model = train_pinyin_to_next_version(train_data_list=dataset_list, jieba_vocabulary=, jieba_freqs=, stop_hours= max_spend_time)



