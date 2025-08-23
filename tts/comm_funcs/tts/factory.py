#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2024-05-17
@author: wangzhongbin
@author: hejunfu
'''

from .edge import Edge
from .xfyun import XfYun
from .volcengine import Volcengine
from .xfyun_g1 import XfYunG1
from .volcengine_g1 import VolcengineG1

class Factory:
    '''
    TTS工厂类
    '''
    @staticmethod
    def get_tts(type:str):
        '''
        获取TTS对象
        @param type: 类型
        '''
        if type == "edge":
            return Edge()
        elif type == "xfyun":
            return XfYun()
        elif type == "volcengine":
            return Volcengine()
        elif type == "xfyun_g1":
            return XfYunG1()
        elif type == "volcengine_g1":
            return VolcengineG1()
        else:
            raise ValueError("Invalid TTS type")