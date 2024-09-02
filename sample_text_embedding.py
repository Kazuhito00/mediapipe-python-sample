#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from typing import List, Any
import time

import cv2  # type:ignore
import mediapipe as mp  # type:ignore
from mediapipe.tasks import python  # type:ignore
from mediapipe.tasks.python import text  # type:ignore

from utils.download_file import download_file


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_text01",
        type=str,
        default="I'm feeling so good",
    )
    parser.add_argument(
        "--input_text02",
        type=str,
        default="I'm okay I guess",
    )

    parser.add_argument(
        "--model",
        type=int,
        choices=[0],
        default=0,
        help='''
        0:Universal Sentence Encoder
        ''',
    )
    parser.add_argument(
        "--unuse_l2_normalize",
        action="store_true",
    )
    parser.add_argument(
        "--use_quantize",
        action="store_true",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args: argparse.Namespace = get_args()

    input_text01: str = args.input_text01
    input_text02: str = args.input_text02

    model: int = args.model
    unuse_l2_normalize: bool = args.unuse_l2_normalize
    use_quantize: bool = args.use_quantize

    use_l2_normalize: bool = not unuse_l2_normalize

    model_url: List[str] = [
        'https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/latest/universal_sentence_encoder.tflite',
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

    # Text Embedder生成
    base_options: python.BaseOptions = python.BaseOptions(
        model_asset_path=model_path)
    options: text.TextEmbedderOptions = text.TextEmbedderOptions(
        base_options=base_options,
        l2_normalize=use_l2_normalize,
        quantize=use_quantize,
    )
    embedder: text.TextEmbedder = text.TextEmbedder.create_from_options(
        options)  # type:ignore

    # 処理時間計測開始
    start_time: float = time.time()

    # 推論実施
    embedding_result01: text.TextEmbedderResult = embedder.embed(input_text01)
    embedding_result02: text.TextEmbedderResult = embedder.embed(input_text02)

    # 類似度計算
    similarity: float = text.TextEmbedder.cosine_similarity(
        embedding_result01.embeddings[0],
        embedding_result02.embeddings[0],
    )

    # 処理時間計測終了
    end_time: float = time.time()
    elapsed_time: int = int((end_time - start_time) * 1000)

    print()
    print('MediaPipe Text Embedding  Demo')
    print('  Input01:', input_text01)
    print('  Input02:', input_text02)
    print('  similarity:', similarity)
    print('  Processing time:', elapsed_time, 'ms')


if __name__ == '__main__':
    main()
