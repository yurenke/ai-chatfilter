from service.main import MainService
from service.twice import TwiceMainService
from service.nickname import NicknameFilter


class MemoryController(object):
    """

    """

    main_service = None
    nickname_filter = None
    remote_twict_service = None

    def __init__(self):
        pass
        

    def get_main_service(self, is_admin = False):
        if self.main_service is None:
            self.main_service = MainService(is_admin)

        return self.main_service

    def get_nickname_filter(self, is_admin = False):
        if self.nickname_filter is None:
            self.nickname_filter = NicknameFilter(is_admin)

        return self.nickname_filter

    def get_remote_twice_service(self):
        if self.remote_twict_service is None:
            self.remote_twict_service = TwiceMainService()

        return self.remote_twict_service



mc = MemoryController()

get_main_service = mc.get_main_service
get_nickname_filter = mc.get_nickname_filter
get_remote_twice_service = mc.get_remote_twice_service