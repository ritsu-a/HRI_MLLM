#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2024-03-08
@author: wangzhongbin
'''


from abc import ABC, abstractmethod

class Protocol(ABC):
    '''
    音频转文字类协议
    '''

    @abstractmethod
    def to_text(self, audio, lang, code, ratio):
        '''
        音频转文字
        @param audio: 音频数据
        @param lang: 语种en英 zh中 mix中英混合
        @param code: 编码 mp3, pcm
        @param ratio: 码率 16k
        @return string: 返回文字，失败返回 None
        '''