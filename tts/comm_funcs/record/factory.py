#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2024-03-08
@author: wangzhongbin
'''

from .vad import VAD
from .button import Button
from typing import Optional, Callable

class Factory:
    '''
    录音工厂类
    '''
    @staticmethod
    def get_record(
        type, 
        on_start_done: Optional[Callable[[], None]] = None
    ):
        '''
        获取录音对象
        @param type: 类型
        '''
        if type == 'vad': # vad
            return VAD(on_start_done)
        if type == 'button': # vad
            return Button(on_start_done)
        else:
            raise ValueError("Invalid Record type")