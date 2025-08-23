#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2024-03-08
@author: wangzhongbin
'''

from .fun import Fun

class Factory:
    '''
    音频转文字工厂类
    '''
    @staticmethod
    def get_asr(type):
        '''
        获取音频转文字对象
        @param type: 类型
        '''
        if type == "fun":
            return Fun()
        else:
            raise ValueError("Invalid ASR type")