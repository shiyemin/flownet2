#!/usr/bin/env python2.7

from __future__ import print_function

import os, numpy as np
import glob
import argparse
import caffe
import tempfile
from math import ceil
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('inputfolder', help='path to the input videos')
parser.add_argument('outputfolder', help='path to the output optical flows')
parser.add_argument('--bound',  help='bound value to truncate the values', default=20, type=int)
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')

args = parser.parse_args()

if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)
if(not os.path.exists(args.inputfolder)): raise BaseException('inputfolder does not exist: '+args.inputfolder)
if(not os.path.exists(args.outputfolder)): raise BaseException('outputfolder does not exist: '+args.outputfolder)

#
# There is some non-deterministic nan-bug in caffe
#
print('Network forward pass using %s.' % args.caffemodel)

lowerBound = -args.bound
upperBound = args.bound

num_blobs = 2
width = -1
height = -1
tmp = None
net = None

def extract_flow(frame_list):
    global width, height, tmp, net
    prev = None
    flow = None
    flow_x = []
    flow_y = []
    iid = 0
    for cur in frame_list:
        iid += 1
        if prev is None:
            prev = cur
            continue
        input_data = []
        if len(prev.shape) < 3: input_data.append(prev[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(prev[np.newaxis, :, :, :].transpose(0, 3, 1, 2))
        if len(cur.shape) < 3: input_data.append(cur[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(cur[np.newaxis, :, :, :].transpose(0, 3, 1, 2))

        if width != input_data[0].shape[3] or height != input_data[0].shape[2] or net is None:
            width = input_data[0].shape[3]
            height = input_data[0].shape[2]

            vars = {}
            vars['TARGET_WIDTH'] = width
            vars['TARGET_HEIGHT'] = height

            divisor = 64.
            vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
            vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

            vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
            vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

            proto = open(args.deployproto).readlines()
            for line in proto:
                for key, value in vars.items():
                    tag = "$%s$" % key
                    line = line.replace(tag, str(value))

                tmp.write(line)

            tmp.flush()

            net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)

        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

        i = 1
        while i<=5:
            i+=1

            net.forward(**input_dict)

            containsNaN = False
            for name in net.blobs:
                blob = net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()

                if has_nan:
                    print('blob %s contains nan' % name)
                    containsNaN = True

            if not containsNaN:
                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')

        flow = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)

        flow_i = 255 * (flow - lowerBound) / (upperBound - lowerBound)
        flow_i = np.int32(np.round(flow_i))
        flow_i = np.maximum(0, np.minimum(255, flow_i))
        flow_x.append(flow_i[:, :, 1])
        flow_y.append(flow_i[:, :, 0])
    return flow_x, flow_y


def save_optical_flow(output_folder, flow_x, flow_y, frame_list):
    nframes = len(flow_x)
    for i in range(nframes):
        iid = i + 1
        out_i = '{0}/flow_i_{1:06d}.jpg'.format(output_folder, iid)
        out_x = '{0}/flow_x_{1:06d}.jpg'.format(output_folder, iid)
        out_y = '{0}/flow_y_{1:06d}.jpg'.format(output_folder, iid)
        cv2.imwrite(out_i, frame_list[i])
        cv2.imwrite(out_x, flow_x[i])
        cv2.imwrite(out_y, flow_y[i])


if not args.verbose:
    caffe.set_logging_disabled()
caffe.set_device(args.gpu)
caffe.set_mode_gpu()

vid_list = glob.glob(args.inputfolder+'/*')
tmp_output_video_dir = ""
print(len(vid_list))
for vid_id, vid_path in enumerate(vid_list):
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(args.outputfolder, vid_name)
    if not os.path.exists(out_full_path):
        print("{} {} processing...".format(vid_id, vid_name))
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass
        frame_list = []
        cap = cv2.VideoCapture(vid_path)
        ret, frame = cap.read()
        while ret:
            frame_list.append(frame)
            ret, frame = cap.read()
        flow_x, flow_y = extract_flow(frame_list)
        save_optical_flow(out_full_path, flow_x, flow_y, frame_list)
        print("{} {} done.".format(vid_id, vid_name))
