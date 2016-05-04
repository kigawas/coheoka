# -*- coding: utf-8 -*-
'''
Stanford CoreNLP wrapper
'''
import json
import requests


class StanfordCoreNLP(object):
    def __init__(self, server_url):
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url

    def annotate(self, text, properties=None):
        if not properties:
            properties = {}
        r = requests.post(self.server_url,
                          params={
                              'properties': str(properties)
                          },
                          data=text)
        output = r.text
        try:
            output = json.loads(output, strict=False)
        except ValueError:
            pass
        return output
