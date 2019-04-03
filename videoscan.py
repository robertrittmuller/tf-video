import tensorflow as tf
import sys
import os
import subprocess
import threading
from multiprocessing import Pool as ThreadPool, TimeoutError, freeze_support
import datetime
import os.path as path
import platform
import argparse
from glob import iglob
from shutil import copy
from io import BytesIO
import PIL
from PIL import Image
import timeit
import uuid
import numpy as np
import psutil
import json

# DONE: Modify detection to only process data after inference has completed.
# DONE: Update smoothing function to work as intended (currently broken)
# DONE: Modify model unpersist function to use loaded model name vs. static assignment.
# DONE: Update FFMPEG to use PIPE function for RGB frames directly into the model.
# TODO: Add support for loading multiple models and performing predictions with all loaded models for every frame and per triggers.
# TODO: Need to cycle through all detected labels and correctly output to report.
# TODO: Add support for basic HTML report which includes processed data & visualizations.

# set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# set start time
start = timeit.default_timer()

parser = argparse.ArgumentParser(description='Process some video files using Tensorflow!')
parser.add_argument('--temppath', '-tp',        dest='temppath',        action='store',     default='./vidtemp/',       help='Path to the directory where temporary files are stored.')
parser.add_argument('--trainingpath', '-rtp',   dest='trainingpath',    action='store',     default='./retraining/',    help='Path to the directory where frames for retraining are stored.')
parser.add_argument('--reportpath', '-rp',      dest='reportpath',      action='store',     default='results/',         help='Path to the directory where results are stored.')
parser.add_argument('--modelpath', '-mp',       dest='modelpath',       action='store',     default='models/',          help='Path to the tensorflow protobuf model file.')
parser.add_argument('--smoothing', '-sm',       dest='smoothing',       action='store',     default='0',                help='Apply a type of "smoothing" factor to detection results.')
parser.add_argument('--fps', '-fps',            dest='fps',             action='store',     default='1',                help='Frames Per Second used to sample input video. ')
parser.add_argument('--height', '-y',           dest='height',          action='store',     default='299',              help='Height of the image frame for processing. ')
parser.add_argument('--width', '-x',            dest='width',           action='store',     default='299',              help='Width of the image frame for processing. ')
parser.add_argument('--traininglower', '-tl',   dest='traininglower',   action='store',     default='40',               help='Lower bound prediction for frame sampling when training flag is set. ')
parser.add_argument('--trainingupper', '-tu',   dest='trainingupper',   action='store',     default='60',               help='Upper bound prediction for frame sampling when training flag is set. ')
parser.add_argument('--allfiles', '-a',         dest='allfiles',        action='store_true',                            help='Process all video files in the directory path.')
parser.add_argument('--deinterlace', '-d',      dest='deinterlace',     action='store_true',                            help='Apply de-interlacing to video frames during extraction.')
parser.add_argument('--outputclips', '-o',      dest='outputclips',     action='store_true',                            help='Output results as video clips containing searched for labelname.')
parser.add_argument('--training', '-tr',        dest='training',        action='store_true',                            help='Saves predicted frames for future model retraining.')
parser.add_argument('--outputpadding', '-op',   dest='outputpadding',   action='store',     default='45',               help='Number of seconds added to the start and end of created video clips.')
parser.add_argument('--keeptemp', '-k',         dest='keeptemp',        action='store_true',                            help='Keep ALL temporary extracted video frames.')
parser.add_argument('--videopath', '-v',        dest='video_path',      action='store',     required=True,              help='Path to video file(s).')

args = parser.parse_args()

def drawProgressBar(percent, barLen=20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()


def percentage(part, whole):
    return 100 * (float(part) / float(whole))

def getinfo(vid_file_path):
    ''' Give a json from ffprobe command line

    @vid_file_path : The absolute (full) path of the video file, string.
    '''
    if type(vid_file_path) != str:
        raise Exception('Gvie ffprobe a full file path of the video')
        return

    command = ["ffprobe",
            "-loglevel",  "quiet",
            "-print_format", "json",
             "-show_format",
             "-show_streams",
             vid_file_path
             ]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = pipe.communicate()
    return json.loads(out)

def copy_files(src_glob, dst_folder):
    for fname in iglob(src_glob):
        newfilename = os.path.basename(fname)
        copy(fname, os.path.join(dst_folder, newfilename))


def remove_video_frames():
    for the_file in os.listdir(video_tempDir):
        file_path = os.path.join(video_tempDir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def save_training_frames(frame, framenumber, label, width, height, score, videofilename):
    # saves the passed RGB values (extracted frame) to a JPEG image with the label name.
    srcpath = os.path.join(args.temppath, '')
    dstpath = os.path.join(args.trainingpath + '/' + label, '')
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    
    # take the passed RGB values and create a JPEG from them.
    image = Image.frombytes('RGB', (int(width), int(height)), frame)
    image.save(dstpath + str(score) + '-' + videofilename + '-' + str(framenumber) + '.jpg')

def decode_video(video_path):
    # launches FFMPEG to decode frames from the video file.
    if args.deinterlace == True:
        deinterlace = 'yadif'
    else:
        deinterlace = ''
    video_filename, video_file_extension = path.splitext(path.basename(video_path))
    print(' ')
    print('Decoding video file ' + video_filename)
    video_temp = os.path.join(video_tempDir, str(video_filename) + '_%04d.jpg')
    command = [
        FFMPEG_PATH, '-i', video_path,
        '-vf', 'fps=' + args.fps, '-q:v', '1', '-vsync', 'vfr', video_temp, '-hide_banner', '-loglevel', '0',
        '-vf', deinterlace, '-vf', 'scale=640x480'
    ]
    subprocess.call(command)

    # Read in the image_data, but sort image paths first because os.listdir results are ordered arbitrarily
    file_paths = [os.path.join(video_tempDir, _) for _ in os.listdir(video_tempDir)]
    file_paths.sort()
    return [tf.gfile.FastGFile(_, 'rb').read() for _ in file_paths if os.path.isfile(_)]


def create_clip(video_path, event, totalframes, videoclipend):
    # creates a video clip of the detected event
    if (event[0] - (int(args.outputpadding) * int(args.fps))) >= 1:
        start = event[0] - int(args.outputpadding) * int(args.fps)
    else:
        start = 1
    if ((int(args.outputpadding) * int(args.fps)) + event[-1]) <= totalframes:
        end = event[-1] + (int(args.outputpadding) * int(args.fps))
    else:
        end = totalframes

    if event[0] >= videoclipend:
        starttime = datetime.timedelta(seconds=(start / int(args.fps)))
        endtime = datetime.timedelta(seconds=(end / int(args.fps)))
        duration = endtime - starttime
        # print('Creating video clip from event which starts at ' + str(starttime) + ' and ends at ' + str(endtime) + '.')

        video_reportDir = os.path.join(args.reportpath, '')
        video_filename, video_file_extension = path.splitext(path.basename(video_path))
        video_clip = video_reportDir + str(video_filename) + '_' + str(start) + '.mp4'

        command = [
            FFMPEG_PATH, '-ss', str(starttime), '-i', str(currentSrcVideo),
            '-t', str(duration), video_clip, '-y', '-loglevel', '0',
        ]
        subprocess.call(command)
        return end
    else:
        return videoclipend


def load_video_filenames(relevant_path):
    included_extenstions = ['avi', 'mp4', 'asf', 'mkv']
    return [fn for fn in os.listdir(relevant_path)
            if any(fn.lower().endswith(ext) for ext in included_extenstions)]


# Loads label files, strips off carriage return
def load_labels(path):
    # load the labels and remove stuff we don't want to display
    file_data = [line.split() for line
                 in tf.gfile.GFile(path)]
    return [item[0].split(":") for item in file_data]

def load_labels_new(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def load_tensor_types(path):
    # reads in the input and output tensors
    with open(path) as file:
        content = file.readlines()
    return content[0].rstrip() + ':0', content[1].rstrip() + ':0'  # return minus any extra characters we don't need


def setup_reporting(passed_filename):
    path = os.path.join(args.reportpath, '')
    reportFileName = path + passed_filename + '_report.csv'
    return open(reportFileName, 'w')


def setup_logging(passed_filename):
    path = os.path.join(args.reportpath, '')
    filename = path + passed_filename + '_results.csv'
    return open(filename, 'w')


def smoothListGaussian(list, strippedXs=False, degree=2):
    window = degree * 2 - 1
    weight = np.array([1.0] * window)
    weightGauss = []
    div_odd = lambda n: (n // 2, n // 2 + 1)

    for i in range(window):
        i = i - degree + 1

        frac = i / float(window)

        gauss = 1 / (np.exp((4 * (frac)) ** 2))

        weightGauss.append(gauss)

    weight = np.array(weightGauss) * weight
    smoothed = [0.0] * (len(list) - window)

    for i in range(len(smoothed)):
        smoothed[i] = float("{0:.4f}".format(sum(np.array(list[i:i + window]) * weight) / sum(weight)))

    padfront, padback = div_odd(window)
    for i in range(0, padfront):
        smoothed.insert(0, 0.0)
    for i in range(0, padback):
        smoothed.append(0.0)

    return smoothed

def load_model(modelpath):
    if os.path.exists(modelpath):
        with tf.gfile.GFile(modelpath, 'rb') as file:
            graph_def1 = tf.GraphDef()
            graph_def1.ParseFromString(file.read())
            tensor_id = str(uuid.uuid4())
            return tensor_id, tf.import_graph_def(graph_def1, name=tensor_id)
    else:
        return False


def write_reports(path, data, smoothing=0):
    smootheddata = []
    logfile = setup_logging(path)
    totalrec = len(data)

    if smoothing > 0:
        # Let's find out which columns have the data vs. just labels.
        prediction_columns = []
        column_count = 0
        for item in data[0]:
            if isinstance(item, float):
                prediction_columns.append(column_count)
            column_count += 1

        # Now lets create new smoothed columns of data!
        for datacolidx in prediction_columns:
            labelcol = []
            label = str(data[0][datacolidx - 1]) + 'SM'
            for i in range(0, totalrec):
                labelcol.append(label)
            smootheddata.append(labelcol)

            newcol = []
            for row in data:
                newcol.append(row[datacolidx])
            newcol = smoothListGaussian(newcol, False, smoothing)
            smootheddata.append(newcol)

    frame_num = 1
    line_num = 0
    for row in data:
        logfile.write('%s, ' % (frame_num))
        for item in row:
            logfile.write('%s, ' % (item))
        if smoothing > 0:
            for smooth_item in smootheddata:
                logfile.write('%s, ' % (smooth_item[line_num]))

        logfile.write("\n")
        frame_num += 1
        line_num += 1

    logfile.close()


def runGraphFaster(video_file_name, input_tensor, output_tensor, labels, session, session_name):
    # Performs inference using the passed model parameters. 
    global flagfound
    global n
    n = 0
    results = []

    # setup pointer to video file
    if args.deinterlace == True:
        deinterlace = 'yadif'
    else:
        deinterlace = ''

    # Let's get the video meta data
    video_filename, video_file_extension = path.splitext(path.basename(video_file_name))
    video_metadata = getinfo(video_file_name)
    num_seconds = int(float(video_metadata['streams'][0]['duration']))
    num_of_frames = int(float(video_metadata['streams'][0]['duration_ts']))
    video_width = int(video_metadata['streams'][0]['width'])
    video_height = int(video_metadata['streams'][0]['height'])
    
    # let's get the real FPS as we don't want duplicate frames!
    effective_fps = int(num_of_frames / num_seconds)
    if effective_fps > int(args.fps):
        effective_fps = int(args.fps)
        num_of_frames = num_seconds * int(args.fps)
    
    source_frame_size = str(video_width) + 'x' + str(video_height)
    target_frame_size = args.width + 'x' + args.height

    if(args.training == True):
        frame_size = source_frame_size
    else:
        frame_size = target_frame_size
        video_width = int(args.width)
        video_height = int(args.height)

    print(' ')
    print('Procesing ' + str(num_seconds) + ' seconds of ' + video_filename + ' at ' + str(effective_fps) + 
        ' frame(s) per second with ' + frame_size + ' source frame size.')
    command = [
        FFMPEG_PATH, '-i', video_file_name,
        '-vf', 'fps=' + args.fps, '-r', args.fps, '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
        '-hide_banner', '-loglevel', '0', '-vf', deinterlace, '-f', 'image2pipe', '-vf', 'scale=' + frame_size, '-'
    ]
    image_pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=4*1024*1024)

    # setup the input and output tensors
    output_tensor = sess1.graph.get_tensor_by_name(session_name + '/' + output_tensor)
    input_tensor = sess1.graph.get_tensor_by_name(session_name + '/' + input_tensor)

    # count the number of labels
    top_k = []
    for i in range(0, len(labels)):
        top_k.append(i)

    while True:
        # read next frame
        raw_image = image_pipe.stdout.read(int(video_width)*int(video_height)*3)
        if not raw_image:
            break # stop processing frames EOF!
        else:
            # Run model and get predictions
            processed_image = np.frombuffer(raw_image, dtype='uint8')
            processed_image = processed_image.reshape((int(video_width), int(video_height), 3))
            
            if frame_size != target_frame_size:
                # we need to fix the frame size so the model does not panic!
                fixed_image = Image.frombytes('RGB', (int(video_width), int(video_height)), processed_image)
                fixed_image = fixed_image.resize((int(args.width), int(args.height)), PIL.Image.ANTIALIAS)
                fixed_image = np.expand_dims(fixed_image, 0)
                final_image = np.divide(np.subtract(fixed_image, [0]), [255])
            else:
                processed_image = processed_image.astype(float)
                processed_image = np.expand_dims(processed_image, 0)
                final_image = np.divide(np.subtract(processed_image, [0]), [255])

            predictions = session.run(output_tensor, {input_tensor: final_image})
            predictions = np.squeeze(predictions)
            image_pipe.stdout.flush()
            n = n + 1

        data_line = []
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]

            score = float("{0:.4f}".format(score))
            data_line.append(human_string)
            data_line.append(score)

            # save frames that are around the decision boundary so they can then be used for later model re-training.
            if args.training == True:
                if score >= float(int(args.traininglower) / 100) and score <= float(int(args.trainingupper) / 100):
                    save_training_frames(raw_image, n, human_string, video_width, video_height, int(score * 100), video_filename)

        results.append(data_line)
        drawProgressBar(percentage(n, (num_of_frames)) / 100, 40)  # --------------------- Start processing logic

    print(' ')
    print(str(n - 1) + ' video frames processed for ' + video_file_name)
    return results

# -- Environment setup
currentSrcVideo = ''

if platform.system() == 'Windows':
    # path to ffmpeg bin
    FFMPEG_PATH = 'ffmpeg.exe'
    FFPROBE_PATH = 'ffprobe.exe'
else:
    # path to ffmpeg bin
    default_ffmpeg_path = '/usr/local/bin/ffmpeg'
    default_ffprobe_path = '/usr/local/bin/ffprobe'
    FFMPEG_PATH = default_ffmpeg_path if path.exists(default_ffmpeg_path) else '/usr/bin/ffmpeg'
    FFPROBE_PATH = default_ffprobe_path if path.exists(default_ffprobe_path) else '/usr/bin/ffprobe'

# setup video temp directory for video frames
video_tempDir = args.temppath
if not os.path.isdir(video_tempDir):
    os.mkdir(video_tempDir)

# get how much ram we have to work with
sysram = psutil.virtual_memory()
sysproc = psutil.cpu_count()

# -- Main processing loop for multiple video files
if __name__ == '__main__':
    # fix for parallel processing
    freeze_support()
    if args.allfiles:
        video_files = load_video_filenames(args.video_path)

        if args.modelpath.endswith('.pb'):
            tensorpath = args.modelpath[:-3] + '-meta.txt'
            labelpath = args.modelpath[:-3] + '-labels.txt'
        else:
            tensorpath = args.modelpath
            labelpath = args.modelpath + '-labels.txt'

        loaded_labels = load_labels_new(labelpath)
        input_tensor, output_tensor = load_tensor_types(tensorpath)
        a_graph_name, a_graph = load_model(args.modelpath)
        sess1 = tf.Session(graph=a_graph)

        for video_file in video_files:
            filename, file_extension = path.splitext(path.basename(video_file))
            n = 0
            flagfound = 0
            remove_video_frames()
            clean_video_path = os.path.join(args.video_path, '')
            currentSrcVideo = clean_video_path + video_file

            output = runGraphFaster(currentSrcVideo, input_tensor, output_tensor, loaded_labels, sess1, a_graph_name)
            write_reports(filename, output, int(args.smoothing))

            # clean up
            image_data = []

    # -- Main processing loop for single video file
    else:
        filename, file_extension = path.splitext(path.basename(args.video_path))
        n = 0
        flagfound = 0
        remove_video_frames()
        currentSrcVideo = args.video_path

        if args.modelpath.endswith('.pb'):
            tensorpath = args.modelpath[:-3] + '-meta.txt'
            labelpath = args.modelpath[:-3] + '-labels.txt'
        else:
            tensorpath = args.modelpath
            labelpath = args.modelpath + '-labels.txt'

        loaded_labels = load_labels(labelpath)
        input_tensor, output_tensor = load_tensor_types(tensorpath)
        a_graph_name, a_graph = load_model(args.modelpath)
        sess1 = tf.Session(graph=a_graph)
        output = runGraphFaster(currentSrcVideo, input_tensor, output_tensor, loaded_labels, sess1, a_graph_name)
        write_reports(filename, output, int(args.smoothing))

    if not args.keeptemp:
        remove_video_frames()

    # -- When main processing has completed, tally up the elapsed time.
    print(' ')
    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)
    sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))
