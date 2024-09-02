#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from typing import List
import time

import numpy as np
from scipy.io import wavfile  # type:ignore
from mediapipe.tasks import python  # type:ignore
from mediapipe.tasks.python.components import containers  # type:ignore
from mediapipe.tasks.python import audio  # type:ignore

from utils.download_file import download_file


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_audio",
        type=str,
        default="asset/hyakuninisshu_02.wav",
    )

    parser.add_argument(
        "--model",
        type=int,
        choices=[0],
        default=0,
        help='''
        0:YamNet
        ''',
    )

    parser.add_argument(
        "--max_results",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args: argparse.Namespace = get_args()

    input_audio_path: str = args.input_audio
    model: int = args.model
    max_results: int = args.max_results

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite',
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

    # Audio Classifier生成
    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: audio.AudioClassifierOptions = audio.AudioClassifierOptions(
        base_options=base_options,
        max_results=max_results,
    )
    classifier: audio.AudioClassifier = audio.AudioClassifier.create_from_options(
        options)  # type:ignore

    # 処理時間計測開始
    start_time: float = time.time()

    # 推論実施
    sample_rate: int
    wav_data: np.ndarray
    sample_rate, wav_data = wavfile.read(input_audio_path)
    audio_clip: containers.AudioData = containers.AudioData.create_from_array(
        wav_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
    classification_result_list: List[
        audio.AudioClassifierResult] = classifier.classify(audio_clip)

    # 処理時間計測終了
    end_time: float = time.time()
    elapsed_time: int = int((end_time - start_time) * 1000)

    # WAVファイルの長さを計算
    duration_seconds = int(len(wav_data) / sample_rate)
    timestamps = [i * 1000 for i in range(duration_seconds)]

    print()
    print('MediaPipe Audio Classification Demo')
    print('  Input:', input_audio_path)
    for index, timestamp in enumerate(timestamps):
        classification_result: audio.AudioClassifierResult = classification_result_list[
            index]
        top_category = classification_result.classifications[0].categories[0]
        print(
            f'    Timestamp {timestamp}: {top_category.category_name} ({top_category.score:.2f})'
        )
    print('  Processing time:', elapsed_time, 'ms')


if __name__ == '__main__':
    main()
