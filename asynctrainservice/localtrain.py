import logging
from temptrainmodel import ChineseChatModel, ChineseNicknameModel

def train():
    model = ChineseNicknameModel()
    model.fit_model()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train()