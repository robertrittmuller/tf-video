# TF-VIDEO
#### Using Tensorflow & TF-SLIM to Detect Features in Video

An automation "wrapper" based on TF-SLIM to make it easy to detect various features in video using Tensorflow, FFMPEG, and various versions of the Inception neural network.

## Features

 - Can utilize virtually any trained model (Inception Resnet V2, Mobilenet, etc) via Tensorflow Hub (retrain.py).
 - Runs under both Linux and Windows (Python 3.6.x).
 - Silly fast, even on CPU for detection. Using an GTX 1060 Nvidia GPU it can perform analysis of a two-hour video in less than 7 minutes even using Inception Resnet V2 and under 3 minutes using Mobilenet V1.

## Requirements

Tensorflow 1.x (tested on 1.7), FFMPEG (Tested on 3.4), and Python 3.6.x (Windows & Linux).
A pre-trained model such as Inception V3 or Mobilenet.

## Installation

No formal "installation" is required beyond making a copy of this directory on your local system.

```bash
$ git clone git@github.com:robertrittmuller/tf-video.git
$ cd tf-video
$ python videoscan.py -h
```

## Commands

### videoscan.py

Search video for features in video. Creates/overwrites `[videofilename]-results.csv`.

#### Simple example:

`videoscan.py` __`--modelpath` models/mymodel.pb `--labelpath` models\mylabelsfilename.txt `--reportpath` ..\example-reports
`--labelname` mylabel [myvideofile.avi]__

#### Complex example:

`videoscan.py` __`--modelpath` models/mymodel.pb `--labelpath` models\mylabelsfilename.txt `--reportpath` ..\example-reports
`--labelname` mylabel `--fps` 5 `--allfiles` `--outputclips` `--smoothing` 2 `--training` --videopath [/path/to/video/files]__

#### Additional Switches & Options

`--modelpath` Path to the tensorflow protobuf model file.
<br>`--modeltype` Type of Tensorflow model being loaded (mobilenetV1, inception_resnet_v2, etc).
<br>`--labelpath` Path to the tensorflow model labels file.
<br>`--labelname` Name of primary label, used to trigger secondary model (if needed).
<br>`--reportpath` Path to the directory where reports/output are stored.
<br>`--temppath` Path to the directory where temporary files are stored.
<br>`--trainingpath` Path to the directory where detected frames for retraining are stored.
<br>`--height` Height of the image frame to be extracted and processed. Needs to match model input layer!
<br>`--width` Width of the image frame to be extracted and processed. Needs to match model input layer!
<br>`--smoothing` Apply a type of "smoothing" factor to detection results. (Range = 1-6)
<br>`--fps` Frames Per Second used to sample input video. The higher this number the slower analysis will go. (Default is 1 FPS)
<br>`--allfiles` Process all video files in the directory path.
<br>`--outputclips` Output results as video clips containing searched for labelname.
<br>`--training` Saves predicted frames for future model retraining.
<br>`--outputpadding` Number of seconds added to the start and end of created video clips.
<br>`--filter` Value used to filter on a label. [Depricated]
<br>`--keeptemp` Keep temporary extracted video frames stored in path specified by `--temppath`

## License

MIT
