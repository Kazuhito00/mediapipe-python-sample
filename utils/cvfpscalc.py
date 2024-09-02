from collections import deque
import cv2 as cv
from typing import Deque


class CvFpsCalc:

    def __init__(self, buffer_len: int = 1) -> None:
        self._start_tick: int = cv.getTickCount()
        self._freq: float = 1000.0 / cv.getTickFrequency()
        self._difftimes: Deque[float] = deque(maxlen=buffer_len)

    def get(self) -> float:
        current_tick: int = cv.getTickCount()
        different_time: float = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps: float = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded: float = round(fps, 2)

        return fps_rounded
