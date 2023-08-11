import xml.etree.ElementTree as ET
import itertools
import re

class XmlParser():

    def __init__(self, file):

        with open(file, 'rb') as f:
            f_byte = f.read()
            f_byte = re.sub(b"[\r\n\&\xa0-\xff]+", b"", f_byte)
            
            xml_bytes = b''.join([b'<root>', f_byte, b'</root>'])
            parser = ET.XMLParser(encoding="utf-8")
            # print('xml_bytes: ', xml_bytes[:1800])
            # check_int = 888800
            # print('xml_bytes[{}:] ~~ '.format(check_int), xml_bytes[check_int:check_int+20])

            root = ET.fromstring(xml_bytes, parser=parser)
            print(root)
            for el in root.findall('review'):
                review_text = el.find('review_text')
                if review_text.text:
                    print(review_text.text)
                    break
        # root = tree.getroot()
        # print(root)

    def readFileAsList(self, path, deep_keys=['review', 'review_text']):
        result_list = []
        with open(path, 'rb') as f:
            f_byte = f.read()
            f_byte = re.sub(b"[\r\n\&\xa0-\xff]+", b"", f_byte)
            
            xml_bytes = b''.join([b'<root>', f_byte, b'</root>'])
            parser = ET.XMLParser(encoding="utf-8")
            # print('xml_bytes: ', xml_bytes[:1800])
            # check_int = 888800
            # print('xml_bytes[{}:] ~~ '.format(check_int), xml_bytes[check_int:check_int+20])

            root = ET.fromstring(xml_bytes, parser=parser)
            print(root)
            for el in root.findall('review'):
                review_text = el.find('review_text')
                if review_text.text:
                    print(review_text.text)
                    break


