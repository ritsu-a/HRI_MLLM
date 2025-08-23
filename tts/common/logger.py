#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-02-12
@author: wangzhongbin
'''

import logging 
import time
import functools
import os
from logging.handlers import TimedRotatingFileHandler

# 创建一个日志记录器
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)  # 设置日志级别
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

if not any(isinstance(h, logging.StreamHandler) for h in LOGGER.handlers):
    logging_stream_handler = logging.StreamHandler()
    logging_stream_handler.setFormatter(formatter)
    LOGGER.addHandler(logging_stream_handler)
if not any(isinstance(h, TimedRotatingFileHandler) for h in LOGGER.handlers):
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs'))
    os.makedirs(log_dir, exist_ok=True)
    time_rotat_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, 'navibot.log'), 
        when='midnight', 
        interval=1, 
        backupCount=7,  # 每天分割
        encoding='utf-8'
    )
    time_rotat_handler.suffix = "%Y-%m-%d.log" 
    time_rotat_handler.setFormatter(formatter)
    LOGGER.addHandler(time_rotat_handler)


def ExecutionTime(description=""):
    '''
    执行耗时装饰器，可传入描述信息
    '''
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            msg = f"{description} {func.__qualname__} 执行耗时: {end_time - start_time:.4f} 秒"
            LOGGER.info(msg)
            return result
        return wrapper
    return decorator

def ExecutionError(func):
    '''
    执行错误装饰器
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            err = f"{func.__qualname__} 执行错误: {str(ex)}"
            LOGGER.error(err)
    return wrapper