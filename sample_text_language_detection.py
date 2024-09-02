#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from typing import List
import time

from mediapipe.tasks import python  # type:ignore
from mediapipe.tasks.python import text  # type:ignore

from utils.download_file import download_file


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_text",
        type=str,
        default="分久必合合久必分",
    )

    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1],
        default=0,
        help='''
        0:Language Detector
        ''',
    )
    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args: argparse.Namespace = get_args()

    input_text: str = args.input_text
    model: int = args.model

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite',
    ]

    # ダウンロードファイル名生成
    model_name: str = model_url[model].split('/')[-1]
    quantize_type: str = model_url[model].split('/')[-3]
    split_name: List[str] = model_name.split('.')
    model_name = split_name[0] + '_' + quantize_type + '.' + split_name[1]

    # 重みファイルダウンロード
    model_path: str = os.path.join('model', model_name)
    if not os.path.exists(model_path):
        download_file(url=model_url[model], save_path=model_path)

    # Language Detector生成
    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: text.LanguageDetectorOptions = text.LanguageDetectorOptions(
        base_options=base_options, )
    detector: text.LanguageDetector = text.LanguageDetector.create_from_options(
        options)  # type:ignore

    # 処理時間計測開始
    start_time: float = time.time()

    # 推論実施
    detection_result: text.LanguageDetectorResult = detector.detect(input_text)

    # 処理時間計測終了
    end_time: float = time.time()
    elapsed_time: int = int((end_time - start_time) * 1000)

    print()
    print('MediaPipe Language Detection Demo')
    print('  Input:', input_text)
    for detection in detection_result.detections:
        print(f'    {detection.language_code}: ({detection.probability:.2f})')
    print('  Processing time:', elapsed_time, 'ms')


if __name__ == '__main__':
    main()
