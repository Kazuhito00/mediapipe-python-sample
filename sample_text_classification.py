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
        default="I'm looking forward to what will come next.",
    )

    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1],
        default=0,
        help='''
        0:BERT-classifier
        1:Average word embedding
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
        'https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/latest/bert_classifier.tflite',
        'https://storage.googleapis.com/mediapipe-models/text_classifier/average_word_classifier/float32/latest/average_word_classifier.tflite',
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

    # Text Classifier生成
    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: text.TextClassifierOptions = text.TextClassifierOptions(
        base_options=base_options, )
    classifier: text.TextClassifier = text.TextClassifier.create_from_options(
        options)  # type:ignore

    # 処理時間計測開始
    start_time: float = time.time()

    # 推論実施
    classification_result: text.TextClassifierResult = classifier.classify(
        input_text)
    top_category = classification_result.classifications[0].categories[0]

    # 処理時間計測終了
    end_time: float = time.time()
    elapsed_time: int = int((end_time - start_time) * 1000)

    print()
    print('MediaPipe Text Classification Demo')
    print('  Input:', input_text)
    print('  Top Category:', top_category.category_name)
    print('  Score:', top_category.score)
    print('  Processing time:', elapsed_time, 'ms')


if __name__ == '__main__':
    main()
