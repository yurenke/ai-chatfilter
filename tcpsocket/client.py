import socket
import os, sys, getopt
from chat_package import pack, unpack
from configparser import RawConfigParser

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
DIR = os.path.dirname(os.path.abspath(__file__))

config_key = RawConfigParser()
config_key.read(DIR+'/keys.cfg')

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = '127.0.0.1'
# host = '172.16.20.120'
port = 8025
bufsize = 1024

argvs = sys.argv[1:]

try:
    opts, args = getopt.getopt(argvs, "h:p:b:")
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

for o, a in opts:
    if o == "-p":
        port = int(a)

    if o == "-h":
        host = str(a)

    if o == '-b':
        bufsize = int(a)

addr = (host, port)

client.connect(addr)

def let_user_pick(options):
    print("Please choose package type:")
    for idx, element in enumerate(options):
        print("{}) {}".format(idx+1,element))
    i = input("Enter number: ")
    try:
        if 0 < int(i) <= len(options):
            return int(i) - 1
    except:
        pass
    return None

cmd_options = ['hearting', 'login', 'login response', 'chatting', 'chatting json', 'chat response', 'nickname filter', 'empty byte']
cmd_ints = [0x000001, 0x040001, 0x040002, 0x040003, 0x041003, 0x040004, 0x040007, 0x000000]

is_keep_chatting = False
msgid = 0
num = 0

while True:

    if is_keep_chatting:
        pass
    else:
        num = let_user_pick(cmd_options)
        if num is None:
            break

    command_hex = cmd_ints[num]
    packed = None

    if command_hex == 0x000001:

        packed = pack(command_hex)

    elif command_hex == 0x040001:

        print("Please enter serverid: ")
        serverid = input()
        print("Please enter sig: ")
        sig = input()
        packed = pack(command_hex, serverid=serverid, sig=sig)

    elif command_hex == 0x040002:

        print("Please enter code: ")
        code = input()
        packed = pack(command_hex, code=int(code))

    elif command_hex == 0x040003:

        if msgid > 0:
            msgid += 1
        else:
            print("Please enter msgid: ")
            msgid = input()
            msgid = int(msgid)
        
        print("Please enter msgtxt: ")
        msgtxt = input()
        if msgtxt:
            packed = pack(command_hex, msgid=msgid, msgtxt=msgtxt)
            is_keep_chatting = True
        else:
            is_keep_chatting = False
            continue

    elif command_hex == 0x041003:

        if msgid > 0:
            msgid += 1
        else:
            print("Please enter msgid: ")
            msgid = input()
            msgid = int(msgid)

        print("Please enter msgtxt: ")
        msgtxt = input()
        print("Please enter roomid: ")
        roomid = input()
        json = {
            'msg': msgtxt,
            'roomid': roomid,
        }
        if msgtxt and roomid:
            packed = pack(command_hex, msgid=msgid, json=json)
            is_keep_chatting = True
        else:
            is_keep_chatting = False
            continue

    elif command_hex == 0x040004:

        print("Please enter msgid: ")
        msgid = input()
        print("Please enter code: ")
        code = input()
        packed = pack(command_hex, msgid=int(msgid), code=int(code))

    elif command_hex == 0x040007:

        print("Please enter nickname: ")

        nickname = input()
        packed = pack(command_hex, reqid=12345, nickname=nickname)
        if nickname:
            is_keep_chatting = True
        else:
            is_keep_chatting = False

    elif command_hex == 0x000000:
        packed = pack()

    
    # if not data or data=='exit':
    #     break
    
    # print('== client == command_hex: ', command_hex)
    # print('== client == packed data: ', packed)
    client.send(packed)

    recv_data = client.recv(bufsize)
    if not recv_data:
        print('--- no recevice ---')
        break

    print('===== receive data =====')
    # print (recv_data.decode('utf-8'))
    trying_unpacked, _ = unpack(recv_data)
    # print(recv_data)
    print(trying_unpacked)
    print('=========================')
    # unpacked_data = unpack('i16s', recv_data)
    # print(unpacked_data)

client.close()
