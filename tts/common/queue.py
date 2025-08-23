#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2025-02-13
@author: wangzhongbin
'''

import threading
from collections import deque

class Queue(threading.Thread):
    '''
    队列，用于业务逻辑隔离
    '''

    def __init__(self):
        '''
        初始化任务队列线程
        '''
        super().__init__()
        self.daemon = True
        self.queue = deque()
        self.cond = threading.Condition()
        self._finished = False

    def finish(self):
        '''
        停止接受任务并清空队列
        '''
        with self.cond:
            self._finished = True
            self.queue.clear()
            self.cond.notify_all()

    def push(self, data):
        '''
        添加任务到队列
        @res data: 数据
        '''
        with self.cond:
            self.queue.append(data)
            self.cond.notify()

    def pop(self):
        '''
        取出队列中的任务，没有任务时等待
        @return 数据
        '''
        with self.cond:
            while len(self.queue) == 0 and not self._finished:
                self.cond.wait()

            if self._finished and len(self.queue) == 0:
                raise StopIteration("Queue is finished and empty.")

            return self.queue.popleft()

