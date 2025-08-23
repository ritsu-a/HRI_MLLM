#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-02-12
@author: wangzhongbin
'''

from .curl_agent import CurlAgent

class Factory:
    '''
    ai工厂方法
    '''
    
    @staticmethod
    def get_agent(type:str, model:str, host:str, api_key:str):
        '''
        获取ai对象
        @param type: 类型
        @param model: 模型
        @param host: url
        @param api_key: key
        @return AI对象
        '''
        if type == 'curl_agent':
            return CurlAgent(model, host, api_key)
        else:
            raise ValueError("Invalid ai type")
