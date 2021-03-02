# mediapipe-python-sample
[MediaPipe](https://github.com/google/mediapipe)のPythonパッケージのサンプルです。<br>
2021/03/03時点でPython実装のある以下6機能について用意しています。
* [Hands](https://google.github.io/mediapipe/solutions/hands)<br>
![suwkm-avmbx](https://user-images.githubusercontent.com/37477845/101514487-a59d8500-39c0-11eb-8346-d3c9ab917ea6.gif)<br>
* [Pose](https://google.github.io/mediapipe/solutions/pose)<br>
![z9e49-wa894](https://user-images.githubusercontent.com/37477845/101512555-7ab23180-39be-11eb-814c-9fad59e0cf9a.gif)<br>
* [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)<br>
![8w6n9-lavib](https://user-images.githubusercontent.com/37477845/101512592-869df380-39be-11eb-8a80-241e272cc195.gif)<br>
* [Holistic](https://google.github.io/mediapipe/solutions/holistic)<br>
![4xbuq-2o9kx](https://user-images.githubusercontent.com/37477845/101908209-1336f480-3bff-11eb-9f3f-5a3055821ebd.gif)<br>
* [Face Detection](https://google.github.io/mediapipe/solutions/face_detection)<br>
![12-01 MediaPipeFaceDetection](https://user-images.githubusercontent.com/37477845/109686899-0e625b00-7bc6-11eb-991e-7fbecfb841cf.gif)<br>
* [Objectron](https://google.github.io/mediapipe/solutions/objectron)<br>
![12-03 MediaPipeObjectron](https://user-images.githubusercontent.com/37477845/109686979-25a14880-7bc6-11eb-8290-4e87968f6044.gif)

# Requirement 
* mediapipe 0.8.3 or later
* OpenCV 3.4.2 or later

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
* --upper_body_only<br>
検出対象を上半身のみにするか ※指定した場合のz座標出力は未対応<br>
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
* --upper_body_only<br>
検出対象を上半身のみにするか ※指定した場合のz座標出力は未対応<br>
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

また、女性の画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の写真を利用しています。
