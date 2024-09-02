# mediapipe-python-sample
[MediaPipe](https://github.com/google/mediapipe)のPythonパッケージのサンプルです。<br>
2021/12/14時点でPython実装のある以下7機能について用意しています。
* [Hands](https://google.github.io/mediapipe/solutions/hands)<br>
<img src="https://user-images.githubusercontent.com/37477845/101514487-a59d8500-39c0-11eb-8346-d3c9ab917ea6.gif" width="45%">　<img src="https://user-images.githubusercontent.com/37477845/146001896-e6a4df4c-7e83-4449-a3af-876491e301ed.gif" width="45%"><br>
* [Pose](https://google.github.io/mediapipe/solutions/pose)<br>
<img src="https://user-images.githubusercontent.com/37477845/101512555-7ab23180-39be-11eb-814c-9fad59e0cf9a.gif" width="45%">　<img src="https://user-images.githubusercontent.com/37477845/126742650-6ab0df29-a2f6-4bb8-8dbc-54db691135e6.gif" width="45%"><br><img src="https://user-images.githubusercontent.com/37477845/130624523-4be1cb41-92b8-4003-a6b5-659ac364a181.gif" width="45%"><br>
* [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)<br>
<img src="https://user-images.githubusercontent.com/37477845/101512592-869df380-39be-11eb-8a80-241e272cc195.gif" width="45%">　<img src="https://user-images.githubusercontent.com/37477845/136427793-bd387581-d3f3-4208-8dea-512f27e6c648.gif" width="45%"><br>
* [Holistic](https://google.github.io/mediapipe/solutions/holistic)<br>
<img src="https://user-images.githubusercontent.com/37477845/101908209-1336f480-3bff-11eb-9f3f-5a3055821ebd.gif" width="45%">　<img src="https://user-images.githubusercontent.com/37477845/126744354-d6307bb2-b720-4e8e-9896-146ee3e7ae94.gif" width="45%"><br><img src="https://user-images.githubusercontent.com/37477845/136389144-6d8ef7cc-e970-4aff-9153-e1bb198c594e.gif" width="45%"><br>
* [Face Detection](https://google.github.io/mediapipe/solutions/face_detection)<br>
<img src="https://user-images.githubusercontent.com/37477845/109686899-0e625b00-7bc6-11eb-991e-7fbecfb841cf.gif" width="45%"><br>
* [Objectron](https://google.github.io/mediapipe/solutions/objectron)<br>
<img src="https://user-images.githubusercontent.com/37477845/109686979-25a14880-7bc6-11eb-8290-4e87968f6044.gif" width="45%"><br>
* [Selfie Segmentation](https://google.github.io/mediapipe/solutions/selfie_segmentation)<br>
<img src="https://user-images.githubusercontent.com/37477845/120812014-8f473f00-c587-11eb-8ac8-944c25c2f264.gif" width="45%"><br>

# Requirement 
* mediapipe 0.8.8 or later<br>※旧バージョンのMediaPipeを使用する場合は[Tags](https://github.com/Kazuhito00/mediapipe-python-sample/tags)の旧コミット版を利用ください
* OpenCV 3.4.2 or later
* matplotlib 3.4.1 or later ※Pose/Holisticでplot_world_landmarkオプションを使用する場合のみ

mediapipeはpipでインストールできます。
```bash
pip install mediapipe
```

# Demo
デモの実行方法は以下です。
#### Face Mesh
```bash
python sample_facemesh.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --max_num_hands<br>
最大手検出数<br>
デフォルト：1
* --refine_landmarks<br>
[ATTENTION MESH MODEL](https://google.github.io/mediapipe/solutions/face_mesh#attention-mesh-model)を使用するか否か ※目と口周りのランドマークがより正確になる<br>
デフォルト：指定なし
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.5
* --min_tracking_confidence<br>
トラッキング信頼値の閾値<br>
デフォルト：0.5
* --use_brect<br>
外接矩形を描画するか否か<br>
デフォルト：指定なし
#### Hands
```bash
python sample_hand.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model_complexity<br>
モデルの複雑度(0:軽量 1:高精度)<br>
デフォルト：1
* --max_num_faces<br>
最大顔検出数<br>
デフォルト：1
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.7
* --min_tracking_confidence<br>
トラッキング信頼値の閾値<br>
デフォルト：0.5
* --use_brect<br>
外接矩形を描画するか否か<br>
デフォルト：指定なし
* --plot_world_landmark<br>
World座標をmatplotlib表示する ※matplotlibを用いるため処理が重くなります<br>
デフォルト：指定なし
#### Pose
```bash
python sample_pose.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model_complexity<br>
モデルの複雑度(0:Lite 1:Full 2:Heavy)<br>
※性能差は[Pose Estimation Quality](https://google.github.io/mediapipe/solutions/pose#pose-estimation-quality)を参照ください<br>
デフォルト：1
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.5
* --min_tracking_confidence<br>
トラッキング信頼値の閾値<br>
デフォルト：0.5
* --enable_segmentation<br>
人物セグメンテーションを有効化するか<br>
デフォルト：指定なし
* --segmentation_score_th<br>
人物セグメンテーションの閾値<br>
デフォルト：0.5
* --use_brect<br>
外接矩形を描画するか否か<br>
デフォルト：指定なし
* --plot_world_landmark<br>
World座標をmatplotlib表示する ※matplotlibを用いるため処理が重くなります<br>
デフォルト：指定なし
#### Holistic
```bash
python sample_holistic.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model_complexity<br>
モデルの複雑度(0:Lite 1:Full 2:Heavy)<br>
※性能差は[Pose Estimation Quality](https://google.github.io/mediapipe/solutions/pose#pose-estimation-quality)を参照ください<br>
デフォルト：1
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.5
* --min_tracking_confidence<br>
トラッキング信頼値の閾値<br>
デフォルト：0.5
* --enable_segmentation<br>
人物セグメンテーションを有効化するか<br>
デフォルト：指定なし
* --unuse_smooth_landmarks<br>
人物セグメンテーションのスムース化を使用しない<br>
デフォルト：指定なし
* --segmentation_score_th<br>
人物セグメンテーションの閾値<br>
デフォルト：0.5
* --use_brect<br>
外接矩形を描画するか否か<br>
デフォルト：指定なし
* --plot_world_landmark<br>
World座標をmatplotlib表示する ※matplotlibを用いるため処理が重くなります<br>
デフォルト：指定なし
#### Face Detection
```bash
python sample_facedetection.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model_selection<br>
モデル選択(0：2m以内の検出に最適なモデル、1：5m以内の検出に最適なモデル)<br>
デフォルト：0
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.5
#### Objectron
```bash
python sample_objectron.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --static_image_mode<br>
静止画像モード ※トラッキング無し<br>
デフォルト：指定なし
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.5
* --min_tracking_confidence<br>
トラッキング信頼値の閾値<br>
デフォルト：0.99
* --model_name<br>
検出対象(20201/03/03時点：'Shoe', 'Chair', 'Cup', 'Camera'の4種類)<br>
デフォルト：Cup
#### Selfie Segmentation
```bash
python sample_selfie_segmentation.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model_selection<br>
モデル種類指定<br>
0：Generalモデル(256x256x1 出力)<br>
1：Landscapeモデル(144x256x1 出力)<br>
デフォルト：0
* --score_th<br>
スコア閾値(閾値以上：人間、閾値未満：背景)<br>
デフォルト：0.1
* --bg_path<br>
背景画像格納パス ※未指定時はグリーンバック<br>
デフォルト：None

# For Raspberry Pi
以下のRaspberry Pi向けビルドを利用することで、Raspberry Pi上で本サンプルを試すことが出来ます。<br>
mediapipe-bin は、v0.8.4 および v0.8.5のバージョンが提供されています。<br>
mediapipe-python-sample は タグv0.8.4、v0.8.5のコードをご使用ください。<br>
* [Raspberry Piで手軽にMediaPipeを楽しむ方法](https://zenn.dev/karaage0703/articles/63fed2a261096d)
* [PINTO0309/mediapipe-bin](https://github.com/PINTO0309/mediapipe-bin)<br><img src="https://user-images.githubusercontent.com/33194443/120130242-a4774300-c200-11eb-8a74-d7f74384a4eb.gif" width="30%">
<!-- [Raspberry PiでMediapipeをPythonで使用する【pipでインストール】](https://www.hiro877.com/entry/rasp-mp-pip-inst) -->


# ToDo
- [x] ~~[Holistic](https://google.github.io/mediapipe/solutions/holistic)のサンプル追加 (mediapipe 0.8.1)~~
- [x] ~~Poseのz座標表示を追加 (mediapipe 0.8.3)~~
- [x] ~~[Face Detection](https://google.github.io/mediapipe/solutions/face_detection)のサンプル追加 (mediapipe 0.8.3)~~
- [x] ~~[Objectron](https://google.github.io/mediapipe/solutions/objectron)のサンプル追加 (mediapipe 0.8.3)~~

# Reference
* [MediaPipe](https://github.com/google/mediapipe)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
mediapipe-python-sample is under [Apache-2.0 License](LICENSE).

また、女性の画像、および背景画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の写真を利用しています。
