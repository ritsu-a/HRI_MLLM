#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-02-12
@author: wangzhongbin
'''

from abc import ABC, abstractmethod
from typing import List, Dict, Callable

class Protocol(ABC):
    '''
    ai类协议
    '''

    @abstractmethod
    def chat(self, messages: List[Dict], stream: bool, callback: Callable[[bool,Dict], None]):
        '''
        chat 推理
        @param messages: 消息 [{'role': 'user', 'content': content}],
        @param stream: 启用流式响应
        @param callback: 回调函数，用于在推理完成后处理结果
        '''
    
    @abstractmethod
    def request(self, messages: List[Dict], stream: bool, callback: Callable[[bool,Dict], None], arguments: Dict = {}, url: str = ''):
        '''
        request 推理 用于发布不同模型请求
        @param messages: 消息 [{'role': 'user', 'content': content}],
        @param stream: 启用流式响应
        @param callback: 回调函数，用于在推理完成后处理结果
        @param arguments: 消息参数，{'temperature':0.6}
        @param url: 接口地址
        '''